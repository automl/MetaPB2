"""
Base code from Ray Tune's Population Based Bandit (PB2) algorithm
(https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.pb2.PB2.html)
described in https://arxiv.org/pdf/2002.02518.pdf
"""
import logging
from copy import deepcopy
from typing import Dict, Optional, Tuple, List, Callable

import numpy as np
import pandas as pd
from ray.tune.experiment import Trial
from scipy.optimize import minimize

from src.algorithms.algorithm_utils import get_indices_to_logtr, process_data, Fixed_TV_SquaredExp, \
    get_fixed_UCB, timing
from src.algorithms.log_pb2 import LogPB2, pb2_gp
from src.algorithms.meta_data import MetaData


def import_pb2_dependencies():
    try:
        import GPy
    except ImportError:
        GPy = None
    try:
        import sklearn
    except ImportError:
        sklearn = None
    return GPy, sklearn


GPy, has_sklearn = import_pb2_dependencies()

if GPy and has_sklearn:
    from GPy.models import GPCoregionalizedRegression
    from ray.tune.schedulers.pb2_utils import (
        normalize,
        optimize_acq,
        select_length,
        standardize,
    )

logger = logging.getLogger(__name__)

@timing
def multi_task_pb2_gp(X, y, current, kernel_kwargs=None):
    if kernel_kwargs is None:
        kernel_kwargs = {}
    kernel_kwargs.setdefault('kernel_type', 'lcm')
    active_dims = list(range(X[0].shape[1]))

    assert kernel_kwargs.get('kernel_type', 'lcm') in ['lcm', 'icm']
    if kernel_kwargs['kernel_type'] == 'lcm':
        kernels_list = []
        for _ in range(len(X)):
            kern = Fixed_TV_SquaredExp(input_dim=X[0].shape[1], active_dims=active_dims, **kernel_kwargs)
            kernels_list.append(kern)
        combined_kern = GPy.util.multioutput.LCM(input_dim=X[0].shape[1], num_outputs=len(X), kernels_list=kernels_list)
    elif kernel_kwargs['kernel_type'] == 'icm':
        combined_kern = GPy.util.multioutput.ICM(input_dim=X[0].shape[1], num_outputs=len(X),
                                                 kernel=Fixed_TV_SquaredExp(X[0].shape[1], **kernel_kwargs))
    m = MultiTaskGP(X, y, kernel=combined_kern)
    m.optimize()

    # m.kern.lengthscale.fix(m.kern.lengthscale.clip(1e-5, 1))

    if current is None:
        m1 = m
    else:
        X[0] = np.vstack((X[0], current))
        ypad = np.zeros(current.shape[0])
        ypad = ypad.reshape(-1, 1)
        y[0] = np.vstack((y[0], ypad))

        if kernel_kwargs['kernel_type'] == 'lcm':
            kernels_list = []
            for _ in range(len(X)):
                kern = Fixed_TV_SquaredExp(input_dim=X[0].shape[1], active_dims=active_dims, **kernel_kwargs)
                kernels_list.append(kern)
            combined_kern = GPy.util.multioutput.LCM(input_dim=X[0].shape[1], num_outputs=len(X),
                                                     kernels_list=kernels_list)
        elif kernel_kwargs['kernel_type'] == 'icm':
            combined_kern = GPy.util.multioutput.ICM(input_dim=X[0].shape[1], num_outputs=len(X),
                                                     kernel=Fixed_TV_SquaredExp(X[0].shape[1], **kernel_kwargs))
        m1 = MultiTaskGP(X, y, kernel=combined_kern)
        m1.optimize()
    return m, m1


class MultiTaskGP(GPCoregionalizedRegression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, X):
        """
        Augments input data such that predictions are only made for the first task.
        """
        X_new = np.hstack([X, np.zeros((X.shape[0], 1))])
        metadata = {'output_index': np.zeros((X.shape[0], 1), dtype=int)}
        return super().predict(X_new, Y_metadata=metadata)

@timing
def optimize_multi_task_acq(func, m, m1, fixed, num_f):
    """Optimize acquisition function."""

    opts = {"maxiter": 200, "maxfun": 200, "disp": False}

    T = 10
    best_value = -999
    best_theta = np.array(m1.X[0, :-1])
    bounds = [(0, 1) for _ in range(m.X.shape[1] - num_f - 1)]

    for ii in range(T):
        x0 = np.random.uniform(0, 1, m.X.shape[1] - num_f - 1)

        res = minimize(
            lambda x: -func(m, m1, x, fixed),
            x0,
            bounds=bounds,
            method="L-BFGS-B",
            options=opts,
        )

        val = func(m, m1, res.x, fixed)
        if val > best_value:
            best_value = val
            best_theta = res.x

    return np.clip(best_theta, 0, 1)

@timing
def select_config(
        Xraw: np.array,
        yraw: np.array,
        meta_data: MetaData,
        current: list,
        newpoint: np.array,
        bounds: dict,
        log_scale_hyperparam: list,
        num_f: int,
        gp_statistics: dict,
        kernel_kwargs: dict = None,
        acquisition_kwargs: dict = None,
):
    """Selects the next hyperparameter config to try.

    This function takes the formatted data, fits the GP model and optimizes the
    UCB acquisition function to select the next point.

    Args:
        Xraw: The un-normalized array of hyperparams, Time and
            Reward
        yraw: The un-normalized vector of reward changes.
        meta_data: Meta data for the meta models.
        current: The hyperparams of trials currently running. This is
            important so we do not select the same config twice. If there is
            data here then we fit a second GP including it
            (with fake y labels). The GP variance doesn't depend on the y
            labels so it is ok.
        newpoint: The Reward and Time for the new point.
            We cannot change these as they are based on the *new weights*.
        bounds: Bounds for the hyperparameters. Used to normalize.
        log_scale_hyperparam: The hyperparameters that are log scaled.
        num_f: The number of fixed params. Almost always 2 (reward+time)
        gp_statistics: Not used.
        kernel_kwargs: keyword arguments for the kernel.
        acquisition_kwargs: Keyword arguments for the acquisition function.

    Return:
        xt: A vector of new hyperparameters.
    """
    if acquisition_kwargs is None:
        acquisition_kwargs = {}

    indices_to_logtr = get_indices_to_logtr(bounds, log_scale_hyperparam) + num_f
    Xraw = deepcopy(Xraw)
    Xraw[:, indices_to_logtr] = np.log(Xraw[:, indices_to_logtr])

    length = select_length(Xraw, yraw, bounds, num_f)

    Xraw = Xraw[-length:, :]
    yraw = yraw[-length:]

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

    fixed = normalize(newpoint, oldpoints)

    if current is not None:
        padding = np.array([fixed for _ in range(current.shape[0])])
        current[:, indices_to_logtr - num_f] = np.log(current[:, indices_to_logtr - num_f])
        current = normalize(current, base_vals)
        current = np.clip(current, 0., 1.)
        current = np.hstack((padding, current))

    # enough datapoints
    min_target_T = np.min(Xraw[:, 0])
    max_target_T = np.max(Xraw[:, 0])
    if max_target_T > min_target_T:
        # toDO: what if the target GP is out of range for the meta GP's?
        num_samples = 100
        weights = meta_data.compute_rank_weights(X, y, Xraw[:, 0], False, num_samples)

        top_k = min(kernel_kwargs.get('top_k', 3), len(meta_data.meta_models))
        closest_models = np.argsort(weights)[-top_k:]  # index of top_k biggest weights
        closest_models = closest_models[weights[closest_models] > 0.]  # filter out weight 0 indices

        combined_X = [X]
        combined_y = [y]
        for idx in closest_models:
            new_time = newpoint[0]

            meta_X, meta_y = meta_data.get_data_slice(idx, new_time, n_points=kernel_kwargs.get('n_points', 100))
            meta_X = meta_data.meta_input_to_target(meta_X, min_target_T, max_target_T, idx)
            meta_y, _ = meta_data.scale_output(meta_y, 0., min_target_T, max_target_T, idx)

            combined_X.append(meta_X)
            combined_y.append(meta_y)

        m, m1 = multi_task_pb2_gp(combined_X, combined_y, current, kernel_kwargs)
        xt = optimize_multi_task_acq(get_fixed_UCB(**acquisition_kwargs), m, m1, fixed, num_f)
    else:
        m, m1 = pb2_gp(X, y, current)  # maybe add kernel kwargs here too? Sort of unncesessary
        xt = optimize_acq(get_fixed_UCB(**acquisition_kwargs), m, m1, fixed, num_f)
    # convert back...
    xt = xt * (np.max(base_vals, axis=0) - np.min(base_vals, axis=0)) + np.min(
        base_vals, axis=0
    )
    xt[indices_to_logtr - num_f] = np.exp(xt[indices_to_logtr - num_f])
    xt = xt.astype(np.float32)

    # todo: save gp statistics
    return xt


def explore(data, meta_data, bounds, log_scale_hyperparam, current, base, old, config, gp_statistics=None,
            kernel_kwargs=None, acquisition_kwargs=None, hp_initializer=None):
    """Returns next hyperparameter configuration to use.

    This function primarily processes the data from completed trials
    and then requests the next config from the select_config function.
    It then adds the new trial to the dataframe, so that the reward change
    can be computed using the new weights.
    It returns the new point and the dataframe with the new entry.
    """
    df = process_data(data, bounds)
    # Only use the last 1k datapoints, so the GP is not too slow.
    df = df.iloc[-1000:, :].reset_index(drop=True)
    # We need this to know the T and Reward for the weights.
    dfnewpoint = df[df["Trial"] == str(base)]

    if not dfnewpoint.empty:
        # Now specify the dataset for the GP.
        y = np.array(df.y.values)
        # Meta data we keep -> episodes and reward.
        # (TODO: convert to curve)
        t_r = df[["Time", "R_before"]]
        hparams = df[bounds.keys()]
        X = pd.concat([t_r, hparams], axis=1).values
        newpoint = df[df["Trial"] == str(base)].iloc[-1, :][["Time", "R_before"]].values
        new = select_config(X, y, meta_data, current, newpoint, bounds, log_scale_hyperparam, num_f=len(t_r.columns),
                            gp_statistics=gp_statistics, kernel_kwargs=kernel_kwargs,
                            acquisition_kwargs=acquisition_kwargs)

        new_config = config.copy()
        values = []
        for i, col in enumerate(hparams.columns):
            if isinstance(config[col], int):
                new_config[col] = int(new[i])
                values.append(int(new[i]))
            else:
                new_config[col] = new[i]
                values.append(new[i])
    else:
        new_config = config.copy()
        if hp_initializer is not None:
            values = hp_initializer()
            for idx, col in enumerate(bounds.keys()):
                new_config[col] = values[idx]
        else:
            values = []
            for col in bounds.keys():
                if isinstance(config[col], int):
                    values.append(int(new_config[col]))
                else:
                    values.append(new_config[col])

    new_T = data[data["Trial"] == str(base)].iloc[-1, :]["Time"]
    new_Reward = data[data["Trial"] == str(base)].iloc[-1, :].Reward

    lst = [[old] + [new_T] + values + [new_Reward]]
    cols = ["Trial", "Time"] + list(bounds) + ["Reward"]
    new_entry = pd.DataFrame(lst, columns=cols)

    # Create an entry for the new config, with the reward from the
    # copied agent.
    data = pd.concat([data, new_entry]).reset_index(drop=True)

    return new_config, data


class MultiTaskPB2(LogPB2):
    """Implements the MultiTaskPB2 algorithm.

    This algorithm is a variant of PB2 that uses meta data to in the form of a multi-task GP.

    Args:
        time_attr: The training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_iteration` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
        metric: The training result objective value attribute. Stopping
            procedures will use this attribute.
        mode: One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute.
        perturbation_interval: Models will be considered for
            perturbation at this interval of `time_attr`. Note that
            perturbation incurs checkpoint overhead, so you shouldn't set this
            to be too frequent.
        hyperparam_bounds: Hyperparameters to mutate. The format is
            as follows: for each key, enter a list of the form [min, max]
            representing the minimum and maximum possible hyperparam values.
        log_scale_hyperparam: List of hyperparameter names that should be
            transformed to be on logscale.
        quantile_fraction: Parameters are transferred from the top
            `quantile_fraction` fraction of trials to the bottom
            `quantile_fraction` fraction. Needs to be between 0 and 0.5.
            Setting it to 0 essentially implies doing no exploitation at all.
        log_config: Whether to log the ray config of each model to
            local_dir at each exploit. Allows config schedule to be
            reconstructed.
        require_attrs: Whether to require time_attr and metric to appear
            in result for every iteration. If True, error will be raised
            if these values are not present in trial result.
        synch: If False, will use asynchronous implementation of
            PBT. Trial perturbations occur every perturbation_interval for each
            trial independently. If True, will use synchronous implementation
            of PBT. Perturbations will occur only after all trials are
            synced at the same time_attr every perturbation_interval.
            Defaults to False. See Appendix A.1 here
            https://arxiv.org/pdf/1711.09846.pdf.
        custom_explore_fn: Custom explore function for perturbing
        kernel_kwargs: Optional arguments to pass to the GP kernel.
        meta_kernel_kwargs: Optional arguments to pass to the meta GP kernel.
        acquisition_kwargs: Optional arguments to pass to the GP acquisition function.
        meta_kwargs: Optional arguments to pass to the MetaData object.
        hp_initializer: Optional function to initialize hyperparameters.
    """

    def __init__(
            self,
            time_attr: str = "time_total_s",
            metric: Optional[str] = None,
            mode: Optional[str] = None,
            perturbation_interval: float = 60.0,
            meta_directory: str = None,
            excluded_metadata: str = None,
            hyperparam_bounds: Dict = None,
            log_scale_hyperparam: List = None,
            quantile_fraction: float = 0.25,
            log_config: bool = True,
            require_attrs: bool = True,
            synch: bool = False,
            custom_explore_fn: Optional[Callable[[dict], dict]] = None,
            kernel_kwargs: Dict = None,
            meta_kernel_kwargs: Dict = None,
            acquisition_kwargs: Dict = None,
            meta_kwargs: Dict = None,
            hp_initializer: Callable = None,
    ):

        gpy_available, sklearn_available = import_pb2_dependencies()
        if not gpy_available:
            raise RuntimeError("Please install GPy to use PB2.")

        if not sklearn_available:
            raise RuntimeError("Please install scikit-learn to use PB2.")

        if meta_directory is None:
            raise ValueError(f'Directory with countaining meta data must be specified')

        super(MultiTaskPB2, self).__init__(
            time_attr=time_attr,
            metric=metric,
            mode=mode,
            perturbation_interval=perturbation_interval,
            hyperparam_bounds=hyperparam_bounds,
            log_scale_hyperparam=log_scale_hyperparam,
            quantile_fraction=quantile_fraction,
            log_config=log_config,
            require_attrs=require_attrs,
            synch=synch,
            custom_explore_fn=custom_explore_fn,
            kernel_kwargs=kernel_kwargs,
            acquisition_kwargs=acquisition_kwargs,
            hp_initializer=hp_initializer,
        )
        if meta_kwargs is None:
            meta_kwargs = {}
        # load metadata and train the models
        self.meta_data = MetaData(meta_directory, self._hyperparam_bounds, self._log_scale_hyperparam,
                                  excluded_metadata, meta_kernel_kwargs, **meta_kwargs)

    def _get_new_config(self, trial: Trial, trial_to_clone: Trial) -> Tuple[Dict, Dict]:
        """Gets new config for trial by exploring trial_to_clone's config using
        Bayesian Optimization (BO) to choose the hyperparameter values to explore.

        Overrides `PopulationBasedTraining._get_new_config`.

        Args:
            trial: The current trial that decided to exploit trial_to_clone.
            trial_to_clone: The top-performing trial with a hyperparameter config
                that the current trial will explore.

        Returns:
            new_config: New hyperparameter configuration (after BO).
            operations: Empty dict since PB2 doesn't explore in easily labeled ways
                like PBT does.
        """
        # If we are at a new timestep, we dont want to penalise for trials
        # still going.
        if self.data["Time"].max() > self.last_exploration_time:
            self.current = None

        new_config, data = explore(
            self.data,
            self.meta_data,
            self._hyperparam_bounds,
            self._log_scale_hyperparam,
            self.current,
            trial_to_clone,
            trial,
            trial_to_clone.config,
            self.gp_statistics,
            self.kernel_kwargs,
            self.acquisition_kwargs,
            self.hp_initializer
        )

        # Important to replace the old values, since we are copying across
        self.data = data.copy()

        # If the current guy being selecting is at a point that is already
        # done, then append the data to the "current" which contains the
        # points in the current batch.
        new = [new_config[key] for key in self._hyperparam_bounds]

        new = np.array(new)
        new = new.reshape(1, new.size)
        if self.data["Time"].max() > self.last_exploration_time:
            self.last_exploration_time = self.data["Time"].max()
            self.current = new.copy()
        else:
            self.current = np.concatenate((self.current, new), axis=0)
            logger.debug(self.current)

        if self._custom_explore_fn:
            new_config = self._custom_explore_fn(new_config)
            assert (
                    new_config is not None
            ), "Custom explore function failed to return a new config"

        return new_config, {}
