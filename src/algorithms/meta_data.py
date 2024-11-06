"""
The weighting code is build on https://github.com/automl/transfer-hpo-framework,
which was introduced the following paper:
Practical Transfer Learning for Bayesian Optimization
Matthias Feurer, Benjamin Letham, Frank Hutter and Eytan Bakshy
https://arxiv.org/pdf/1802.02219v3.pdf
"""

import glob

import numpy as np
import pandas as pd
from ray.tune.schedulers.pb2_utils import standardize, normalize
from sklearn.metrics import mean_squared_error

from src.algorithms.algorithm_utils import get_indices_to_logtr, timing
from src.algorithms.log_pb2 import pb2_gp


def roll_col(X: np.ndarray, shift: int) -> np.ndarray:
    """
    Rotate columns to right by shift.
    """
    return np.concatenate((X[:, -shift:], X[:, :-shift]), axis=1)


def transform_meta_data(Xraw, yraw, bounds, log_scale_hyperparam):
    num_f = 2
    indices_to_logtr = get_indices_to_logtr(bounds, log_scale_hyperparam) + num_f
    Xraw[:, indices_to_logtr] = np.log(Xraw[:, indices_to_logtr])
    base_vals = np.array(list(bounds.values())).T
    base_vals[:, indices_to_logtr - num_f] = np.log(base_vals[:, indices_to_logtr - num_f])
    oldpoints = Xraw[:, :num_f]
    old_lims = np.concatenate(
        (np.max(oldpoints, axis=0), np.min(oldpoints, axis=0))
    ).reshape(2, oldpoints.shape[1])
    limits = np.concatenate((old_lims, base_vals), axis=1)
    X = normalize(Xraw, limits)
    X[:, num_f:] = np.clip(X[:, num_f:], 0., 1.)
    y = standardize(yraw).reshape(yraw.size, 1)
    return X, y

def select_elements_around_index(lst, index_to_center_around, number_of_elements):
    start_index = max(0, index_to_center_around - int(number_of_elements / 2))
    end_index = min(len(lst), start_index + number_of_elements)
    start_index = max(0, end_index - number_of_elements)
    selected_elements = lst[start_index:end_index]
    return selected_elements

class MetaData:
    @timing
    def __init__(self, meta_directory, hyperparam_bounds, log_scale_hyperparam, excluded_metadata=None,
                 kernel_kwargs=None, loss_function='ranking', all_data=True, fraction=2, num_meta=None):
        assert loss_function in ['ranking', 'stratified_ranking', 'mse', 'ges']
        self.loss_function = loss_function
        meta_directory = '/'.join(meta_directory.split('/')[:-1])
        meta_models = []
        file_path_regex = meta_directory + r'/*/*.csv'
        env_names = [schedule_file.split('/')[-1].removesuffix('.csv') for schedule_file in glob.glob(file_path_regex)]
        env_names = list(set(env_names))
        for env_name in env_names:
            # Do not learn from the current evironment
            if excluded_metadata is not None and env_name in excluded_metadata:
                continue
            schedule = []
            dataset_regex = meta_directory + r'/*' + f'/{env_name}.csv'
            for dataset_schedule_file in glob.glob(dataset_regex):
                schedule.append(pd.read_csv(dataset_schedule_file))

            schedule = pd.concat(schedule, axis=0).reset_index(drop=True)
            # extract from there
            schedule = schedule.sort_values(by="Time").reset_index(drop=True)
            yraw = np.array(schedule.y.values)
            t_r = schedule[["Time", "R_before"]]
            hparams = schedule[hyperparam_bounds.keys()]
            Xraw = pd.concat([t_r, hparams], axis=1).values
            X, y = transform_meta_data(Xraw, yraw, hyperparam_bounds, log_scale_hyperparam)
            m, _ = pb2_gp(X, y, None, kernel_kwargs)

            meta_models.append({
                "name": env_name,
                "model": m,
                "meta_X": X,
                "meta_y": y,
                "meta_tr_raw": Xraw[:, :2],
                "meta_yraw": yraw,
            })
            # until here / return new_model and append it to meta_models
        if len(meta_models) == 0:
            raise ValueError(
                f"The given meta directory does not contain any *_schedule.csv files. Directory path: {meta_directory}")
        self.meta_models = meta_models
        self.weights_over_time = pd.DataFrame(
            columns=['time_step', *[model['name'] for model in meta_models], 'target'])


    def target_input_to_meta(self, X, min_target_T, max_target_T, model_index):
        """
        Scales the input data from target task to match the scale of the model specified by model index.
        """
        X = X.copy()  # we don't want to change the target task data

        full_tr = self.meta_models[model_index]["meta_tr_raw"]
        full_min = np.min(full_tr, axis=0)
        full_max = np.max(full_tr, axis=0)

        matching_tr = full_tr[np.logical_and(full_tr[:, 0] <= max_target_T, full_tr[:, 0] >= min_target_T)]
        matching_min = np.min(matching_tr, axis=0)
        matching_max = np.max(matching_tr, axis=0)

        X[:, :2] = (X[:, :2] * (matching_max - matching_min + 1e-8) + matching_min - full_min) / (
                full_max - full_min + 1e-8)
        return X

    def meta_input_to_target(self, X, min_target_T, max_target_T, model_index):
        """
        Scales input from meta task to match target task.
        """
        X = X.copy()  # we don't want to change the meta task data

        full_tr = self.meta_models[model_index]["meta_tr_raw"]
        full_min = np.min(full_tr, axis=0)
        full_max = np.max(full_tr, axis=0)

        matching_tr = full_tr[np.logical_and(full_tr[:, 0] <= max_target_T, full_tr[:, 0] >= min_target_T)]
        matching_min = np.min(matching_tr, axis=0)
        matching_max = np.max(matching_tr, axis=0)

        X[:, :2] = (X[:, :2] * (full_max - full_min + 1e-8) + full_min - matching_min) / (
                matching_max - matching_min + 1e-8)

        if min_target_T == max_target_T:  # normalization quirk for when we only have 1 time value
            X[:, 0] = X[:, 0] * 1e-9

        return X

    def get_data_slice(self, model_index: int, time_point: float, n_points: int = 200):
        """
        This method assumes that the data is ordered by time
        """
        time_point = int(time_point)
        full_time = self.meta_models[model_index]["meta_tr_raw"][:, 0].astype(int)

        center_index = np.asarray(full_time == time_point).nonzero()[0].min()
        lower_bound = max(0, center_index - int(n_points / 2))
        upper_bound = min(len(full_time), n_points + lower_bound)
        lower_bound = max(0, upper_bound - n_points)

        X = self.meta_models[model_index]["meta_X"][lower_bound:upper_bound]
        y = self.meta_models[model_index]["meta_y"][lower_bound:upper_bound]
        return X, y

    def scale_output(self, y, var, min_target_T, max_target_T, model_index):
        """
        Scales output from mata task specified by model_index to match target task.
        """
        full_y = self.meta_models[model_index]["meta_yraw"]
        mean_full = np.mean(full_y, axis=0)
        std_full = np.std(full_y, axis=0) + 1e-8

        full_tr = self.meta_models[model_index]["meta_tr_raw"]
        matching_y = full_y[np.logical_and(full_tr[:, 0] <= max_target_T, full_tr[:, 0] >= min_target_T)]
        matching_mean = np.mean(matching_y, axis=0)
        matching_std = np.std(matching_y, axis=0) + 1e-8

        transformed_y = (y * std_full + mean_full - matching_mean) / matching_std  # should I clip this?
        transformed_var = var * (std_full / matching_std) ** 2
        return transformed_y, transformed_var

    @timing
    def compute_rank_weights(
            self,
            train_x: np.ndarray,
            train_y: np.ndarray,
            raw_time: np.ndarray,
            include_target: bool,
            num_samples: int,
    ) -> np.ndarray:
        """
        Compute ranking weights for each base model and the target model
        (using LOOCV for the target model).
        Returns
        -------
        weights : np.ndarray
        """

        """
        Different test size than trainsize was removed since it had no impact 
        and made everything unnecessarily complicated
        """
        test_size = len(train_x)
        test_x = train_x[-test_size:]
        test_y = train_y[-test_size:]
        test_raw_time = raw_time[-test_size:]

        predictions = []
        min_target_T = np.min(raw_time)
        max_target_T = np.max(raw_time)
        for model_index, model_data in enumerate(self.meta_models):
            model = model_data["model"]
            transformed_x = self.target_input_to_meta(test_x, min_target_T, max_target_T, model_index)
            model_prediction = model.predict(transformed_x)[0].flatten()
            model_prediction = self.scale_output(model_prediction, 0., min_target_T, max_target_T, model_index)[0]
            predictions.append(model_prediction)

        train_masks = np.eye(test_size, len(train_x), k=len(train_x) - test_size, dtype=bool)
        test_masks = np.eye(test_size, dtype=bool)
        train_x_cv = np.stack([train_x[~m] for m in train_masks])
        train_y_cv = np.stack([train_y[~m] for m in train_masks])
        test_x_cv = np.stack([test_x[m] for m in test_masks])

        if include_target:
            loo_prediction = []
            for i in range(test_y.shape[0]):
                m, _ = pb2_gp(train_x_cv[i], train_y_cv[i], None)
                loo_prediction.append(m.predict(test_x_cv[i])[0][0][0])  # how to handle
            predictions.append(loo_prediction)

        predictions = np.array(predictions)

        if self.loss_function == 'ges':
            rank_weights = greedy_ensemble_search(test_y, predictions, num_samples)
            if include_target:
                self.weights_over_time.loc[len(self.weights_over_time)] = [max_target_T, *rank_weights]
            else:
                self.weights_over_time.loc[len(self.weights_over_time)] = [max_target_T, *rank_weights, np.nan]
            return rank_weights

        bootstrap_indices = np.random.choice(predictions.shape[1],
                                             size=(num_samples, predictions.shape[1]),
                                             replace=True)

        bootstrap_predictions = []
        bootstrap_targets = test_y[bootstrap_indices].reshape((num_samples, len(test_y)))
        time_strata = test_raw_time[bootstrap_indices]

        for m in range(predictions.shape[0]):  # len predictions
            bootstrap_predictions.append(predictions[m, bootstrap_indices])

        ranking_losses = np.zeros((predictions.shape[0], num_samples))
        for i in range(len(self.meta_models)):
            if self.loss_function == 'ranking':
                ranking_losses[i] = calculate_ranking_loss(bootstrap_predictions[i], bootstrap_predictions[i],
                                                           bootstrap_targets)
            elif self.loss_function == 'stratified_ranking':
                ranking_losses[i] = calculate_stratified_ranking_loss(bootstrap_predictions[i],
                                                                      bootstrap_predictions[i],
                                                                      bootstrap_targets, time_strata)
            elif self.loss_function == 'mse':
                ranking_losses[i] = calculate_mse(bootstrap_targets, bootstrap_predictions[i])
        if include_target:
            if self.loss_function == 'ranking':
                ranking_losses[-1] = calculate_ranking_loss(bootstrap_predictions[-1], bootstrap_targets,
                                                            bootstrap_targets)
            elif self.loss_function == 'stratified_ranking':
                ranking_losses[i] = calculate_stratified_ranking_loss(bootstrap_predictions[-1], bootstrap_targets,
                                                                      bootstrap_targets, time_strata)
            elif self.loss_function == 'mse':
                ranking_losses[-1] = calculate_mse(bootstrap_targets, bootstrap_predictions[i])

        # compute best model (minimum ranking loss) for each sample
        # this differs from v1, where the weight is given only to the target model in case of a tie.
        # Here, we distribute the weight fairly among all participants of the tie.
        minima = np.min(ranking_losses, axis=0)
        assert len(minima) == num_samples
        best_models = np.zeros(predictions.shape[0])  # make this a dimension of ranking loss or so
        for i, minimum in enumerate(minima):
            minimum_locations = ranking_losses[:, i] == minimum
            sample_from = np.where(minimum_locations)[0]

            for sample in sample_from:
                best_models[sample] += 1. / len(sample_from)

        # compute proportion of samples for which each model is best
        rank_weights = best_models / num_samples
        if include_target:
            self.weights_over_time.loc[len(self.weights_over_time)] = [max_target_T, *rank_weights]
        else:
            self.weights_over_time.loc[len(self.weights_over_time)] = [max_target_T, *rank_weights, np.nan]
        return rank_weights


def calculate_mse(targets, predictions):
    ranking_loss = []
    for j in range(targets.shape[0]):
        ranking_loss.append(mean_squared_error(targets[j], predictions[j]))
    return np.array(ranking_loss)


def calculate_ranking_loss(predictions_left, predictions_right, targets):
    ranking_loss = 0.
    for j in range(targets.shape[1]):
        ranking_loss += np.sum((roll_col(predictions_left, j) < predictions_right) ^ (roll_col(targets, j) < targets),
                               axis=1)
    return ranking_loss


def calculate_stratified_ranking_loss(predictions_left, predictions_right, targets, stratas):
    ranking_losses = []

    for i in range(targets.shape[0]):
        ranking_loss = 0.
        bootstrap_stratas = np.unique(stratas[i])
        for strata in bootstrap_stratas:
            mask = stratas[i] == strata
            m_left = predictions_left[i, mask]
            m_right = predictions_right[i, mask]
            m_target = targets[i, mask]
            for j in range(len(m_target)):
                ranking_loss += np.sum((np.roll(m_left, j) < m_right) ^ (np.roll(m_target, j) < m_target))
        ranking_losses.append(ranking_loss)
    return np.array(ranking_losses)


def greedy_ensemble_search(target, predictions, iterations):
    weights = np.zeros((1, len(predictions)))
    for i in range(iterations):
        loss = np.inf
        index = 0
        for j in range(len(predictions)):
            pred = (weights @ predictions + predictions[j]) / (np.sum(weights) + 1.)
            new_loss = mean_squared_error(target.flatten(), pred.flatten())
            if new_loss < loss:
                index = j
                loss = new_loss
        weights[0, index] += 1

    return weights.flatten() / np.sum(weights)
