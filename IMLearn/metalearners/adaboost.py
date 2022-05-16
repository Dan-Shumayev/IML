from typing import Callable, NoReturn

import numpy as np

from ..base import BaseEstimator
from ..metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # TODO - understand it!

        self.models_ = list()
        self.weights_ = np.zeros(self.iterations_)

        m = X.shape[0]
        self.D_ = np.zeros((self.iterations_, m))  # Weights to be updated by AdaBoost
        self.D_[0, :] = np.ones(m) / m  # Initial distri. is uniform

        for t in range(self.iterations_):
            self.models_.append(self.wl_())
            t_prediction = self.models_[t].fit(X, self.D_[t, :] * y).predict(X)

            epsilon = np.sum(self.D_[t, :] * (1 - (y == t_prediction)))
            self.weights_[t] = 0.5 * np.log(1.0 / epsilon - 1)

            if t < self.iterations_ - 1:
                self.D_[t+1, :] = self.D_[t, :] * np.exp(-self.weights_[t] * y * t_prediction)
                self.D_[t+1, :] /= np.sum(self.D_[t+1, :])  # Normalize weights

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, self.iterations_) # Predict over all iterations

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
        return self.partial_loss(X, y, self.iterations_) # Loss under all the iterations

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        y_hat = np.zeros(X.shape[0])

        T_weights = self.weights_[:T]
        T_models = self.models_[:T]
        
        for model_weight, model in zip(T_weights, T_models):
            y_hat += model_weight * model.predict(X)

        return np.sign(y_hat)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.partial_predict(X, T))
