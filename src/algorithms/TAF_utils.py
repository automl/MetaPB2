import numpy as np
from scipy.stats import norm

def get_expected_improvement(eta=0, **kwargs):
    def expected_improvement(m, m1, x, fixed):
        xtest = np.concatenate((fixed.reshape(-1, 1), np.array(x).reshape(-1, 1))).T

        try:
            preds = m.predict(xtest)
            mean = preds[0][0][0]
        except ValueError:
            mean = -9999

        try:
            preds = m1.predict(xtest)
            std = np.sqrt(preds[1][0][0])
        except ValueError:
            std = 1

        curr_inc = np.max(m.Y)  # check if this makes sense

        Z = (mean - curr_inc - eta) / std
        cdf = norm.cdf(Z)
        pdf = norm.pdf(Z)
        ei = std * (Z * cdf + pdf)
        return max(ei, 0)
    return expected_improvement


# TODO change num_fixed to be passed from class
def get_TAF(x_train, meta_data, weights, t_range, eta=0, past_window=None, **kwargs):
    y_max = []
    if past_window is not None and t_range[1] != t_range[0]:
        min_t = (t_range[1] - past_window - t_range[0]) / (t_range[1] - t_range[0] + 1e-8)
        filter = x_train[:, 0] >= min_t
        x_train = x_train[filter]

    for idx, meta_model in enumerate(meta_data.meta_models):
        model = meta_model["model"]
        x_train_transformed = meta_data.target_input_to_meta(x_train, t_range[0], t_range[1], idx)
        y_pred = model.predict(x_train_transformed)[0]
        y_pred = meta_data.scale_output(y_pred, 0., t_range[0], t_range[1], idx)[0]
        y_max.append(np.max(y_pred))

    def TAF(m, m1, x, fixed):
        # calculate expected improvement of x according to current take
        expected_improvement = get_expected_improvement(eta)
        current_task_ei = expected_improvement(m, m1, x, fixed)
        # calculate the improvement according to meta tasks
        improvement = [current_task_ei * weights[-1]]
        x_test = np.concatenate((fixed.reshape(-1, 1), np.array(x).reshape(-1, 1))).T
        for idx, meta_model in enumerate(meta_data.meta_models):
            if weights[idx] == 0.:
                continue

            model = meta_model["model"]
            # get mean
            x_transformed = meta_data.target_input_to_meta(x_test, t_range[0], t_range[1], idx)
            mean = model.predict(x_transformed)[0]
            mean = meta_data.scale_output(mean, 0., t_range[0], t_range[1], idx)[0][0][0]

            # add the improvement to the array
            improvement.append(max(mean - y_max[idx], 0.) * weights[idx])

        taf_acq_value = sum(improvement)

        return taf_acq_value
    return TAF
