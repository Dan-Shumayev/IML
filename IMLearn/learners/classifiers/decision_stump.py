from __future__ import annotations

from itertools import product
from typing import NoReturn, Tuple

import numpy as np

from ...base import BaseEstimator
from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> None:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        minimum_error: float = np.inf  # Initial error to begin with

        # Find threshold for each vector feature against each sign, and pick the minimal
        # over all the thresholds
        for curr_feature_ix, sign in product(range(X.shape[1]), [-1, 1]):
            threshold, error = self._find_threshold(X[:, curr_feature_ix], y, sign)

            if error < minimum_error: # Pick the minimum error
                self.sign_, self.threshold_, self.j_ = sign, threshold, curr_feature_ix
                minimum_error = error

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        assert self.sign_, "Must be defined!"

        res = list()
        for x in X[:, self.j_]:
            res.append(self.sign_) if x >= self.threshold_ else res.append(-self.sign_)

        return np.array(res)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # TODO - understand it!

        m = values.shape[0]

        labels_to_values = np.zeros((2, m))
        labels_to_values[0, :], labels_to_values[1, :] = labels, values
        # Sort it so that we can iterate over the thresholds in O(N)
        labels_to_values = labels_to_values[:, labels_to_values[1, :].argsort()]
        threshold = labels_to_values[1, 0]

        curr_error = min_error = np.sum(np.abs(labels_to_values[0, :]) * (1 - (sign * np.ones(m) == np.sign(labels_to_values[0, :]))))

        for ix in range(m):
            if np.sign(labels_to_values[0, ix]) == sign:
                curr_error += np.abs(labels_to_values[0, ix])
            else:
                curr_error -= np.abs(labels_to_values[0, ix])

            if curr_error < min_error:
                min_error = curr_error
                threshold = labels_to_values[1, m - 1] + 1 if ix > m - 2 else labels_to_values[1, ix + 1]

        return threshold, min_error


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self._predict(X))

