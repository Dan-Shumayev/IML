from __future__ import annotations

import sys

from sklearn.metrics import mean_squared_error

sys.path.append('c:\\Users\\fserv\\Downloads\\Year 4\\Semester B\\IML\\IML-Projects')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IMLearn.learners.regressors import (LinearRegression, PolynomialFitting,
                                         RidgeRegression)
from IMLearn.metrics import mean_square_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test
from sklearn import datasets
from sklearn.linear_model import Lasso
from utils import *


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best 
    fitting degree.

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for 
    # eps Gaussian noise and split into training- and testing portions
    dataset = x = np.linspace(-1.2, 2, n_samples)
    f_x = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    eps = np.random.normal(0, noise, n_samples)

    responses = f_x + eps

    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(dataset), pd.Series(responses), 2/3)

    go.Figure([go.Scatter(name='Train Error', x=train_x[0], y=train_y, mode='markers', marker_color='rgb(152,171,150)'), 
            go.Scatter(name='Test Error', x=test_x[0], y=test_y, mode='markers', marker_color='rgb(25,115,132)'),
            go.Scatter(name='True Model', x=x, y=f_x, mode='markers', marker_color='rgb(25,115,100)')])\
        .update_layout(title=r"$\text{Polynomial function}$", 
                    xaxis_title=r"$x$", 
                    yaxis_title=r"$\text{y}$").show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    max_deg = 10
    degrees = np.arange(max_deg + 1)

    train_errors = np.empty(max_deg + 1)
    validation_errors = np.empty(max_deg + 1)

    for deg in degrees:
        train_score, validation_score = cross_validate(PolynomialFitting(deg), \
                        np.array(train_x), np.array(train_y).ravel(), mean_square_error)
        
        train_errors[deg] = train_score
        validation_errors[deg] = validation_score

    go.Figure([go.Scatter(name='Training Error', x=degrees, y=train_errors, \
        mode='markers+lines', marker_color='rgb(152,171,150)'), 
            go.Scatter(name='Validation Error', x=degrees, y=validation_errors, \
                mode='markers+lines', marker_color='rgb(25,115,132)')])\
        .update_layout(title=r"$\text{5-Fold Cross-Validation for Polynomial fitting}$", 
                    xaxis_title=r"$\text{Polynomial Degree}$", 
                    yaxis_title=r"$\text{Error}$").show()

    # Question 3 - Using best value of k, 
    # fit a k-degree polynomial model and report test error
    print()

    min_valid_err_deg = np.argmin(validation_errors)
    
    print(f"Num of samples={n_samples} ; noise={noise}")
    print(f"Degree of the polynomial with minimal validation error: {min_valid_err_deg}")
    
    fitted_best_model = PolynomialFitting(min_valid_err_deg).fit(dataset, responses)
    
    print(fr"Best model's loss: {fitted_best_model.loss(np.array(test_x) if test_x.shape[1] != 1 else np.array(test_x).ravel(), np.array(test_y))}")

def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting 
    regularization parameter values for Ridge and Lasso regressions.

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)

    # Split the dataset to a training and testing portion
    train_size = n_samples

    train_X, train_y = X[:train_size, :], y[:train_size]
    test_X, test_y = X[train_size:, :], y[train_size:]

    # Question 7 - Perform CV for different values of the regularization parameter for 
    # Ridge and Lasso regressions
    n_evaluations = 500
    k_fold = 5

    # Ridge regression
    ridge_lambdas = np.linspace(1e-3, 1e1, num=n_evaluations)
    ridge_train_errors = np.empty(len(ridge_lambdas))
    ridge_validation_errors = np.empty(len(ridge_lambdas))

    for i, lam in enumerate(ridge_lambdas):
        train_error, validation_error = cross_validate(RidgeRegression(lam), train_X, train_y,\
            mean_square_error, k_fold) 

        ridge_train_errors[i] = train_error
        ridge_validation_errors[i] = validation_error

    go.Figure([go.Scatter(name='Training Error', x=ridge_lambdas, y=ridge_train_errors, \
    mode='markers+lines', marker_color='rgb(152,171,150)'), 
        go.Scatter(name='Validation Error', x=ridge_lambdas, y=ridge_validation_errors, \
            mode='markers+lines', marker_color='rgb(25,115,132)')])\
    .update_layout(title=r"$\text{5-Fold Cross-Validation for Ridge regression}$", 
                xaxis_title=r"$\text{Regularization term}$", 
                yaxis_title=r"$\text{Error}$").show()

    print()
    ridge_min_valid_err_idx = np.argmin(ridge_validation_errors)
    print(f"Corresponding regularization term (Ridge): {ridge_lambdas[ridge_min_valid_err_idx]}")


    # Lasso regression
    lasso_lambdas = np.linspace(1e-5, 1e-1, n_evaluations)
    lasso_train_errors = np.empty(len(lasso_lambdas))
    lasso_validation_errors = np.empty(len(lasso_lambdas))

    for i, lam in enumerate(lasso_lambdas):
        train_error, validation_error = cross_validate(Lasso(lam, tol=1e-9, max_iter=100000), train_X, train_y,\
            mean_square_error, k_fold) 

        lasso_train_errors[i] = train_error
        lasso_validation_errors[i] = validation_error

    go.Figure([go.Scatter(name='Training Error', x=lasso_lambdas, y=lasso_train_errors, \
                    mode='markers+lines', marker_color='rgb(152,171,150)'), 
                go.Scatter(name='Validation Error', x=lasso_lambdas, y=lasso_validation_errors, \
                    mode='markers+lines', marker_color='rgb(25,115,132)')])\
    .update_layout(title=r"$\text{5-Fold Cross-Validation for Lasso regression}$", 
                xaxis_title=r"$\text{Regularization term}$", 
                yaxis_title=r"$\text{Error}$").show()

    print()
    lasso_min_valid_err_idx = np.argmin(lasso_validation_errors)
    print(f"Corresponding regularization term (Lasso): {lasso_lambdas[lasso_min_valid_err_idx]}")

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model

    ridge_fitted_best_model = RidgeRegression(ridge_lambdas[ridge_min_valid_err_idx]).fit(train_X, train_y)
    lasso_fitted_best_model = Lasso(lasso_lambdas[lasso_min_valid_err_idx], tol=1e-9, \
        max_iter=100000).fit(train_X, train_y)
    lr_model = LinearRegression().fit(train_X, train_y)

    def LS_error(X, y, model) -> float:
        return mean_square_error(y, model.predict(X))

    print()
    print(f"Error for Linear Regression:\t {LS_error(test_X, test_y, lr_model)}")
    print(f"Error for Ridge:\t {LS_error(test_X, test_y, ridge_fitted_best_model)}")
    print(f"Error for Lasso:\t {LS_error(test_X, test_y, lasso_fitted_best_model)}")

if __name__ == '__main__':
    np.random.seed(0)

    # Part 1:
    select_polynomial_degree()  # 1.1-1.3
    select_polynomial_degree(100, 0)  # 1.4
    select_polynomial_degree(1500, 10)  # 1.5

    # Part 2:
    select_regularization_parameter()
