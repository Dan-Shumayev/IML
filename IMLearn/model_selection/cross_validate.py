from __future__ import annotations

from copy import deepcopy
from typing import Callable, Tuple

import numpy as np
from IMLearn import BaseEstimator

# from sklearn.metrics import mean_squared_error


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) \
                       -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for 
        each sample and potentially additional arguments. 
        The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    m = X.shape[0]
    fold_size = m // cv
    remainder = m % cv

    # Randomly partition the dataset to k disjoint subsets:
    shuffled_sample_indices = np.random.permutation(m)
    X_shuffled = X[shuffled_sample_indices]
    y_shuffled = y[shuffled_sample_indices]

    # Accommodate the train- and validation- errors for each fold:
    train_errors = np.empty(cv)
    validation_errors = np.empty(cv)

    for i in range(cv):
        if remainder > 0:
            remainder -= 1
            j = 1
        else:
            j = 0

        num_samples_before_subset_i = fold_size * i + j

        train_indices = np.concatenate([np.arange(0, num_samples_before_subset_i),
                                        np.arange(num_samples_before_subset_i + fold_size, m)])
        validation_indices = np.arange(num_samples_before_subset_i, 
                                        num_samples_before_subset_i + fold_size)

        train_x, validation_x = X_shuffled[train_indices, :] if X_shuffled.shape[1] != 1 \
                                                else X_shuffled[train_indices, :].ravel(), \
                                X_shuffled[validation_indices, :] if X_shuffled.shape[1] != 1 \
                                    else X_shuffled[validation_indices, :].ravel()
        
        train_y, validation_y = y_shuffled[train_indices], y_shuffled[validation_indices]

        model = estimator.fit(train_x, train_y)

        train_y_predicted = model.predict(train_x)
        validation_y_predicted = model.predict(validation_x)

        train_errors[i] = scoring(train_y, train_y_predicted)
        validation_errors[i] = scoring(validation_y, validation_y_predicted)

    mean_train_error = train_errors.mean()
    mean_validation_error = validation_errors.mean()

    return mean_train_error, mean_validation_error
