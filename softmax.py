"""Softmax regression functions."""

import numpy as np
import scipy.optimize as optim

from scipy.misc import logsumexp


class SoftmaxRegression:
    """Models the conditional distribution of discrete outcomes."""

    def __init__(self, k, p, l2, add_intercept=True):
        self._num_out = k
        self._num_in = p
        self._penalty = l2
        self._intercept = add_intercept
        self._weights = None

    @property
    def num_in(self):
        """The number of predictors in the model."""
        return self._num_in + self._intercept

    @property
    def num_out(self):
        """The number of outcomes predicted by the model."""
        return self._num_out

    @property
    def weights(self):
        """The weights parameterizing the model."""
        return np.array(self._weights)

    def predict(self, X):
        """Predict conditional probabilities of outcomes.

        Parameters
        ----------
        X : Predictor matrix.

        Returns
        -------
        A matrix of conditional probabilities. Each row corresponds to
        an input row in the predictor matrix and each column
        corresponds to an outcome.

        """
        if self._weights is None:
            raise RuntimeError('The model has not been fit.')

        if self._intercept:
            X = self._add_intercept(X)

        scores = X @ self._weights.T
        log_probs = scores - logsumexp(scores, axis=1)[:, None]

        return np.exp(log_probs)

    def fit(self, Y, X):
        """Fit the softmax model to data.

        Parameters
        ----------
        Y : Response matrix.
        X : Predictor matrix.

        Returns
        -------
        The fit model.

        """
        if self._intercept:
            X = self._add_intercept(X)

        solution = _learn_weights(Y, X, self._penalty)
        self._weights = solution['x'].reshape((self.num_out, self.num_in))
        return self

    @staticmethod
    def _add_intercept(X):
        """Add a column of ones to the predictors."""
        return np.c_[np.ones(len(X)), X]


def _learn_weights(Y, X, penalty, seed=0):
    """Fit the weights of a multinomial regression model.

    Parameters
    ----------
    Y : Response matrix.
    X : Predictor matrix.

    Returns
    -------
    A scipy optimization solution object. The learned weights are
    indexed by 'x'.

    Notes
    -----
    The rows of the weight _matrix_ are outcome-continuous. To pack
    back into matrix form, you can use:

    > solution['x'].reshape((num_out, num_in))

    """
    _, num_out, num_in = _validate_input(Y, X)

    func = lambda w: -_log_likelihood(w, Y, X) + penalty / 2 * w @ w
    grad = lambda w: -_log_likelihood_grad(w, Y, X) + penalty * w

    rng = np.random.RandomState(seed)
    weights = rng.normal(size=num_out * num_in)
    solution = optim.minimize(func, weights, method='BFGS', jac=grad)

    return solution


def _log_likelihood(weights, Y, X):
    """Log-likelihood of the regression model.

    Parameters
    ----------
    weights : A vector of model weights.
    Y : Response matrix.
    X : Predictor matrix.

    Returns
    -------
    Log-likelihood of the observed data using the given model weights.

    """
    num_obs, num_out, _ = _validate_input(Y, X)

    logl = 0.0

    for i in range(num_obs):
        z = np.kron(np.eye(num_out), X[i]) @ weights
        logl += Y[i] @ _log_softmax(z)

    return logl


def _log_likelihood_grad(weights, Y, X):
    """Gradient of the regression log-likelihood.

    Parameters
    ----------
    weights : A vector of model weights.
    Y : Response matrix.
    X : Predictor matrix.

    Returns
    -------
    A vector with the same length as weights.

    """
    num_obs, num_out, _ = _validate_input(Y, X)

    grad = np.zeros(len(weights))

    for i in range(num_obs):
        A = np.kron(np.eye(num_out), X[i])
        G = _log_softmax_jac(A @ weights)
        grad += A.T @ G.T @ Y[i]

    return grad


def _log_softmax(x):
    """Log-softmax of a vector."""
    return x - logsumexp(x)


def _log_softmax_jac(x):
    """Gradient of the log-softmax.

    Notes
    -----
    This function returns the _jacobian_, and so each row of the
    matrix corresponds to the partial derivatives of each input with
    respect to the outputs.

    """
    p = np.exp(_log_softmax(x))
    return np.eye(len(x)) - np.tile(p, (4, 1))


def _validate_input(Y, X):
    """Check and return the problem dimensions.

    Parameters
    ----------
    Y : Response matrix.
    X : Predictor matrix.

    Returns
    -------
    The number of observations, the number of output classes, and the
    number of predictors.

    """
    num_y, num_out = Y.shape
    num_x, num_in = X.shape

    assert num_y == num_x

    num_obs = num_y

    return num_obs, num_out, num_in
