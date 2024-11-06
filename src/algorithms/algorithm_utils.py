"""
Code fixes some issues with the Ray implementation of the kernels and acquisition functions.
"""

from functools import wraps
from time import time

import numpy as np
from GPy.core import Param
from GPy.kern import Kern
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result

    return wrap


def process_data(data, bounds):
    """Returns dataframe which is used as input for GP.

    This function processes the data from completed trials
    and returns them in a form that can be used as input for gps
    """
    df = data.sort_values(by="Time").reset_index(drop=True)
    df = df.astype({'Trial': 'str'})
    # Group by trial ID and hyperparams.
    # Compute change in timesteps and reward.
    df["y"] = df.groupby(["Trial"] + list(bounds.keys()))["Reward"].diff()
    df["t_change"] = df.groupby(["Trial"] + list(bounds.keys()))["Time"].diff()

    # Delete entries without positive change in t.
    df = df[df["t_change"] > 0].reset_index(drop=True)
    df["R_before"] = df.Reward - df.y

    # Normalize the reward change by the update size.
    # For example if trials took diff lengths of time.
    df["y"] = df.y / df.t_change
    df = df[~df.y.isna()].reset_index(drop=True)
    df = df.sort_values(by="Time").reset_index(drop=True)
    return df


def get_indices_to_logtr(bounds, log_scale_hyperparam):
    indices_to_logtr = np.array([]).astype(int)
    i = 0  # we start indexing with the Hyperparameter columns
    for key in bounds.keys():
        if (key in log_scale_hyperparam):
            # we want to log transform this
            indices_to_logtr = np.append(indices_to_logtr, i)
        i = i + 1
    return indices_to_logtr


def get_fixed_UCB(exploration=0.2, **kwargs):
    def fixed_UCB(m, m1, x, fixed):
        """UCB acquisition function. Interesting points to note:
        1) We concat with the fixed points, because we are not optimizing wrt
           these. This is the Reward and Time, which we can't change. We want
           to find the best hyperparameters *given* the reward and time.
        2) We use m to get the mean and m1 to get the variance. If we already
           have trials running, then m1 contains this information. This reduces
           the variance at points currently running, even if we don't have
           their label.
           Ref: https://jmlr.org/papers/volume15/desautels14a/desautels14a.pdf

        """

        c1 = exploration
        c2 = 0.4
        beta_t = c1 + max(0, np.log(c2 * m.X.shape[0]))
        kappa = np.sqrt(beta_t)

        xtest = np.concatenate((fixed.reshape(-1, 1), np.array(x).reshape(-1, 1))).T

        try:
            preds = m.predict(xtest)
            mean = preds[0][0][0]
        except ValueError:
            mean = -9999

        try:
            preds = m1.predict(xtest)
            var = preds[1][0][0]
        except ValueError:
            var = 0
        return mean + kappa * np.sqrt(max(var, 0))

    return fixed_UCB


def fixed_UCB(m, m1, x, fixed):
    """UCB acquisition function. Interesting points to note:
    1) We concat with the fixed points, because we are not optimizing wrt
       these. This is the Reward and Time, which we can't change. We want
       to find the best hyperparameters *given* the reward and time.
    2) We use m to get the mean and m1 to get the variance. If we already
       have trials running, then m1 contains this information. This reduces
       the variance at points currently running, even if we don't have
       their label.
       Ref: https://jmlr.org/papers/volume15/desautels14a/desautels14a.pdf

    """

    c1 = 0.2
    c2 = 0.4
    beta_t = c1 + max(0, np.log(c2 * m.X.shape[0]))
    kappa = np.sqrt(beta_t)

    xtest = np.concatenate((fixed.reshape(-1, 1), np.array(x).reshape(-1, 1))).T

    try:
        preds = m.predict(xtest)
        mean = preds[0][0][0]
    except ValueError:
        mean = -9999

    try:
        preds = m1.predict(xtest)
        var = preds[1][0][0]
    except ValueError:
        var = 0
    return mean + kappa * np.sqrt(max(var, 0))


class Fixed_TV_SquaredExp(Kern):
    """Time varying squared exponential kernel.
    For more info see the TV-GP-UCB paper:
    http://proceedings.mlr.press/v51/bogunovic16.pdf
    """

    def __init__(
            self, input_dim, variance=1.0, lengthscale=1.0, epsilon=0.0, active_dims=None, lengthscale_ub=1.,
            epsilon_ub=0.5, ARD=False, seperate_y=False, **kwargs
    ):
        super().__init__(input_dim, active_dims, "time_se")
        self.variance = Param("variance", variance)
        self.variance.constrain_bounded(0.01, 100.)
        self.epsilon = Param("epsilon", epsilon)
        if ARD:
            self.lengthscale = []
            for i in range(input_dim - 1):
                self.lengthscale.append(Param(f"lengthscale_{i}", lengthscale))
            self.link_parameters(self.variance, *self.lengthscale, self.epsilon)
            for len_param in self.lengthscale:
                len_param.constrain_bounded(1e-5, lengthscale_ub)
        elif seperate_y:
            self.lengthscale = Param("lengthscale", lengthscale)
            self.y_lengthscale = Param("y_lengthscale", lengthscale)
            self.link_parameters(self.variance, self.lengthscale, self.epsilon, self.y_lengthscale)
            self.lengthscale.constrain_bounded(1e-5, lengthscale_ub)
            self.y_lengthscale.constrain_bounded(1e-5, lengthscale_ub)  # maybe change it to 1.
        else:
            self.lengthscale = Param("lengthscale", lengthscale)
            self.link_parameters(self.variance, self.lengthscale, self.epsilon)
            self.lengthscale.constrain_bounded(1e-5, lengthscale_ub)

        self.epsilon.constrain_bounded(0., epsilon_ub)

    def K(self, X, X2):
        if X2 is None:
            X2 = np.copy(X)
        else:
            if X2.shape[1] != self.input_dim:
                print('fixed!')
                X2 = X2[:, self.active_dims]
        T1 = X[:, 0].reshape(-1, 1)
        T2 = X2[:, 0].reshape(-1, 1)
        dists = pairwise_distances(T1, T2, "cityblock")
        timekernel = (1 - self.epsilon) ** (0.5 * dists)

        X = X[:, 1:]
        X2 = X2[:, 1:]
        if isinstance(self.lengthscale, list):
            dist2 = np.zeros_like(timekernel)
            for i, length in enumerate(self.lengthscale):
                dist2 += np.square(pairwise_distances(X[:, i].reshape(-1, 1), X2[:, i].reshape(-1, 1))) / length
            RBF = self.variance * np.exp(-dist2)
        elif hasattr(self, 'y_lengthscale'):
            dist2 = np.square(euclidean_distances(X[:, 1:], X2[:, 1:])) / self.lengthscale + \
                    np.square(pairwise_distances(X[:, 0].reshape(-1, 1), X2[:, 0].reshape(-1, 1))) / self.y_lengthscale
            RBF = self.variance * np.exp(-dist2)
        else:
            RBF = self.variance * np.exp(-np.square(euclidean_distances(X, X2)) / self.lengthscale)

        return RBF * timekernel

    def Kdiag(self, X):
        return self.variance * np.ones(X.shape[0])

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None:
            X2 = np.copy(X)
        T1 = X[:, 0].reshape(-1, 1)
        T2 = X2[:, 0].reshape(-1, 1)
        n = pairwise_distances(T1, T2, "cityblock") / 2
        timekernel = (1 - self.epsilon) ** n

        X = X[:, 1:]
        X2 = X2[:, 1:]
        if isinstance(self.lengthscale, list):
            dist2 = np.zeros_like(timekernel)
            for i, length in enumerate(self.lengthscale):
                dist2 += np.square(pairwise_distances(X[:, i].reshape(-1, 1), X2[:, i].reshape(-1, 1))) / length
        elif hasattr(self, 'y_lengthscale'):
            dist2 = euclidean_distances(X[:, 1:], X2[:, 1:], squared=True) / self.lengthscale + \
                    np.square(pairwise_distances(X[:, 0].reshape(-1, 1), X2[:, 0].reshape(-1, 1))) / self.y_lengthscale
        else:
            dist2 = euclidean_distances(X, X2, squared=True) / self.lengthscale

        dvar = np.exp(-dist2) * timekernel
        self.variance.gradient = np.sum(dvar * dL_dK)

        deps = -n * (1 - self.epsilon) ** (n - 1) * self.variance * np.exp(-dist2)
        self.epsilon.gradient = np.sum(deps * dL_dK)

        if isinstance(self.lengthscale, list):
            for i, length in enumerate(self.lengthscale):
                deriv = np.square(pairwise_distances(X[:, i].reshape(-1, 1), X2[:, i].reshape(-1, 1))) / (length ** 2)
                dl = timekernel * self.variance * np.exp(-dist2) * deriv
                length.gradient = np.sum(dl * dL_dK)
        elif hasattr(self, 'y_lengthscale'):
            dl = euclidean_distances(X[:, 1:], X2[:, 1:], squared=True) / (self.lengthscale ** 2)
            dyl = np.square(pairwise_distances(X[:, 0].reshape(-1, 1), X2[:, 0].reshape(-1, 1))) / self.y_lengthscale
            dl *= timekernel * self.variance * np.exp(-dist2)
            dyl *= timekernel * self.variance * np.exp(-dist2)
            self.lengthscale.gradient = np.sum(dl * dL_dK)
            self.y_lengthscale.gradient = np.sum(dyl * dL_dK)
        else:
            dl = timekernel * self.variance * np.exp(-dist2) * dist2 / self.lengthscale
            self.lengthscale.gradient = np.sum(dl * dL_dK)

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = np.sum(dL_dKdiag)

    def gradients_X(self, dL_dK, X, X2):
        if X2 is None:
            X2 = np.copy(X)
        T1 = X[:, 0].reshape(-1, 1)
        T2 = X2[:, 0].reshape(-1, 1)
        n = pairwise_distances(T1, T2, "cityblock") / 2
        timekernel = (1 - self.epsilon) ** n

        X = X[:, 1:]
        X2 = X2[:, 1:]
        if isinstance(self.lengthscale, list):
            raise NotImplementedError()
            # dist = np.zeros_like(timekernel)
            # for i, length in enumerate(self.lengthscale):
            #     dist += np.square(pairwise_distances(X[:, i].reshape(-1, 1), X2[:, i].reshape(-1, 1))) / length
        elif hasattr(self, 'y_lengthscale'):
            raise NotImplementedError()
            # dist = euclidean_distances(X[:, 1:], X2[:, 1:], squared=True) / self.lengthscale + \
            #        np.square(pairwise_distances(X[:, 0].reshape(-1, 1), X2[:, 0].reshape(-1, 1))) / self.y_lengthscale
        else:
            dist = euclidean_distances(X, X2, squared=True) / self.lengthscale

        diff = np.expand_dims(X, axis=1) - np.expand_dims(X2, axis=0)
        dx = self.variance / self.lengthscale * 2 * np.sum(
            np.expand_dims(timekernel * np.exp(-dist) * dL_dK, axis=2) * diff, axis=1)

        dt = 0.5 * self.variance * np.sum(np.sign(n) * np.log(1 - self.epsilon) * timekernel * np.exp(-dist) * dL_dK,
                                          axis=1)
        return np.concatenate((np.expand_dims(dt, axis=1), dx), axis=1)

    def gradients_X_diag(self, dL_dKdiag, X):
        # no diagonal gradients
        pass


class Unconstrained_Fixed_TV_SquaredExp(Kern):
    """Time varying squared exponential kernel.
    For more info see the TV-GP-UCB paper:
    http://proceedings.mlr.press/v51/bogunovic16.pdf
    """

    def __init__(
            self, input_dim, variance=1.0, lengthscale=1.0, epsilon=0.0, active_dims=None, ARD=False, seperate_y=False,
            **kwargs
    ):
        super().__init__(input_dim, active_dims, "time_se")
        self.variance = Param("variance", variance)
        self.epsilon = Param("epsilon", epsilon)
        if ARD:
            self.lengthscale = []
            for i in range(input_dim - 1):
                self.lengthscale.append(Param(f"lengthscale_{i}", lengthscale))
            self.link_parameters(self.variance, *self.lengthscale, self.epsilon)
        elif seperate_y:
            self.lengthscale = Param("lengthscale", lengthscale)
            self.y_lengthscale = Param("y_lengthscale", lengthscale)
            self.link_parameters(self.variance, self.lengthscale, self.epsilon, self.y_lengthscale)
        else:
            self.lengthscale = Param("lengthscale", lengthscale)
            self.link_parameters(self.variance, self.lengthscale, self.epsilon)

    def K(self, X, X2):
        # time must be in the far left column
        if self.epsilon > 0.5:
            self.epsilon = 0.5
        elif self.epsilon < 0.:
            self.epsilon = 0.

        if self.variance <= 0:
            self.variance = 0.1

        if X2 is None:
            X2 = np.copy(X)
        else:
            if X2.shape[1] != self.input_dim:
                print('fixed!')
                X2 = X2[:, self.active_dims]
        T1 = X[:, 0].reshape(-1, 1)
        T2 = X2[:, 0].reshape(-1, 1)
        dists = pairwise_distances(T1, T2, "cityblock")
        timekernel = (1 - self.epsilon) ** (0.5 * dists)

        X = X[:, 1:]
        X2 = X2[:, 1:]
        if isinstance(self.lengthscale, list):
            dist2 = np.zeros_like(timekernel)
            for i, length in enumerate(self.lengthscale):
                dist2 += np.square(pairwise_distances(X[:, i].reshape(-1, 1), X2[:, i].reshape(-1, 1))) / length
            RBF = self.variance * np.exp(-dist2)
        elif hasattr(self, 'y_lengthscale'):
            dist2 = np.square(euclidean_distances(X[:, 1:], X2[:, 1:])) / self.lengthscale + \
                    np.square(pairwise_distances(X[:, 0].reshape(-1, 1), X2[:, 0].reshape(-1, 1))) / self.y_lengthscale
            RBF = self.variance * np.exp(-dist2)
        else:
            RBF = self.variance * np.exp(-np.square(euclidean_distances(X, X2)) / self.lengthscale)

        return RBF * timekernel

    def Kdiag(self, X):
        return self.variance * np.ones(X.shape[0])

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None:
            X2 = np.copy(X)
        T1 = X[:, 0].reshape(-1, 1)
        T2 = X2[:, 0].reshape(-1, 1)
        n = pairwise_distances(T1, T2, "cityblock") / 2
        timekernel = (1 - self.epsilon) ** n

        X = X[:, 1:]
        X2 = X2[:, 1:]
        if isinstance(self.lengthscale, list):
            dist2 = np.zeros_like(timekernel)
            for i, length in enumerate(self.lengthscale):
                dist2 += np.square(pairwise_distances(X[:, i].reshape(-1, 1), X2[:, i].reshape(-1, 1))) / length
        elif hasattr(self, 'y_lengthscale'):
            dist2 = np.square(euclidean_distances(X[:, 1:], X2[:, 1:])) / self.lengthscale + \
                    np.square(pairwise_distances(X[:, 0].reshape(-1, 1), X2[:, 0].reshape(-1, 1))) / self.y_lengthscale
        else:
            dist2 = np.square(euclidean_distances(X, X2)) / self.lengthscale

        dvar = np.exp(-dist2) * timekernel
        self.variance.gradient = np.sum(dvar * dL_dK)

        deps = -n * (1 - self.epsilon) ** (n - 1) * self.variance * np.exp(-dist2)
        self.epsilon.gradient = np.sum(deps * dL_dK)

        if isinstance(self.lengthscale, list):
            for i, length in enumerate(self.lengthscale):
                deriv = np.square(pairwise_distances(X[:, i].reshape(-1, 1), X2[:, i].reshape(-1, 1))) / (length ** 2)
                dl = timekernel * self.variance * np.exp(-dist2) * deriv
                length.gradient = np.sum(dl * dL_dK)
        elif hasattr(self, 'y_lengthscale'):
            dl = np.square(euclidean_distances(X[:, 1:], X2[:, 1:])) / (self.lengthscale ** 2)
            dyl = np.square(pairwise_distances(X[:, 0].reshape(-1, 1), X2[:, 0].reshape(-1, 1))) / self.y_lengthscale
            dl *= timekernel * self.variance * np.exp(-dist2)
            dyl *= timekernel * self.variance * np.exp(-dist2)
            self.lengthscale.gradient = np.sum(dl * dL_dK)
            self.y_lengthscale.gradient = np.sum(dyl * dL_dK)
        else:
            dl = timekernel * self.variance * np.exp(-dist2) * dist2 / self.lengthscale
            self.lengthscale.gradient = np.sum(dl * dL_dK)

    def clip_lengthscales(self, upper_bound):
        if isinstance(self.lengthscale, list):
            for lengthscale in self.lengthscale:
                lengthscale.fix(lengthscale.clip(1e-5, upper_bound))
        elif hasattr(self, 'y_lengthscale'):
            self.y_lengthscale.fix(self.y_lengthscale.clip(1e-5, upper_bound))
        else:
            self.lengthscale.fix(self.lengthscale.clip(1e-5, upper_bound))


def sampler(portfolio, hyperparameter):
    """

    Args:
        portfolio: list of dicts in portfolio
        hyperparameter: which hyperparameter do we want to sample

    Returns:
        value of hyperparameter in portfolio

    """
    new_list_of_dicts = [{k: v for k, v in p.items() if k == hyperparameter} for p in portfolio]
    iterator = iter(new_list_of_dicts)

    def sample_function():
        return next(iterator)[hyperparameter]

    return sample_function
