import sys

sys.path.append('C:\\Users\\fserv\\Downloads\\Year 4\\Semester B\\IML\\IML-Projects')

from typing import Callable, List, Tuple, Type

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IMLearn import BaseModule
from IMLearn.desent_methods import ExponentialLR, FixedLR, GradientDescent
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics.loss_functions import mean_square_error
from IMLearn.model_selection.cross_validate import cross_validate
from IMLearn.utils import split_train_test
from sklearn.metrics import auc, mean_squared_error, roc_curve


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] 
        over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, \
                    density=70, showscale=False), \
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], \
                        mode="markers+lines", marker_color="black")], \
                     layout=go.Layout(xaxis=dict(range=xrange), yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], \
                        List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, 
    recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, 
        recording the objective's value and parameters at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    weights, losses = [], []

    def callback(solver=None, weight=None, val=None, grad=None, t=None, eta=None, delta=None):
        weights.append(weight)
        losses.append(val)

    return callback, weights, losses

def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    out_type = "best"

    fig = go.Figure()

    for eta in etas:
        curr_eta = FixedLR(eta)
        l1_callback, l1_weights, l1_losses = get_gd_state_recorder_callback()
        l2_callback, l2_weights, l2_losses = get_gd_state_recorder_callback()

        # Initialize L1-2 norms
        l1_model = L1(init)
        l2_model = L2(init)

        # Set gradient descent algorithms for both norms with `curr_eta` learning-rate
        l1_solver, l2_solver = GradientDescent(learning_rate=curr_eta, out_type=out_type, \
            callback=l1_callback), GradientDescent(learning_rate=curr_eta, \
                                                out_type=out_type, callback=l2_callback)

        # Fit these norms by gradient descent iterations (done by `fit`)
        lowest_loss_l1, lowest_loss_l2 = l1_solver.fit(f=l1_model, X=np.empty, y=np.empty), \
            l2_solver.fit(f=l2_model, X=np.empty, y=np.empty)
        
        print(f"The Lowest Loss Obtained for the L1-norm for eta={eta} is {lowest_loss_l1}.\n")
        print(f"The Lowest Loss Obtained for the L2-norm for eta={eta} is {lowest_loss_l2}.\n")

        fig.add_trace(go.Scatter(x=l1_losses, y=np.arange(len(l1_losses)), \
            text=f"L1-Norm recording with eta={eta}"))
        fig.add_trace(go.Scatter(x=l2_losses, y=np.arange(len(l2_losses)), \
            text=f"L2-Squared Norm recording with eta={eta}"))
        
        if eta == .01:
            plot_descent_path(module=L1,
                              descent_path=np.array(l1_weights),
                              title="L1-norm Gradient Descent Path").show()
            plot_descent_path(module=L2,
                              descent_path=np.array(l2_weights),
                              title="L2-norm Gradient Descent Path").show()

    fig.show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of 
    # the exponentially decaying learning rate
    fig = go.Figure()

    apply_l1_norm = lambda lr: np.linalg.norm(lr, ord=1)

    out_type="best"

    best_gamma = .0
    min_lr = float('inf')

    for gamma in gammas:
        callback, _ , losses = get_gd_state_recorder_callback()
        solver = GradientDescent(learning_rate=ExponentialLR(eta, gamma), \
            out_type=out_type, callback=callback)

        best_value = solver.fit(L1(init), X=np.empty((0, 0)), y=np.empty(0))
        l1_norm = apply_l1_norm(best_value)
        min_lr, best_gamma = (l1_norm, gamma) if l1_norm < min_lr else (min_lr, best_gamma)

        print(f"L1-norm achieves a loss of {best_value} with (eta={eta},gamma={gamma}).\n")

        fig.add_trace(go.Scatter(x=losses, y=np.arange(len(losses)), \
            text=f"Decay Learning-Rate for (eta={eta},gamma={gamma})"))

    print(f"Lowest L1 loss={min_lr} achieved with (eta={eta},gamma={best_gamma}).\n")

    # Plot algorithm's convergence for the different values of gamma
    fig.update_layout(title_text="L1-Norm recording for Exponential Decaying Learning Rate")
    fig.update_xaxes(title_text="Iteration Number")
    fig.update_yaxes(title_text="Loss value")
    fig.show()

    # Plot descent path for gamma=0.95
    l1_callback, l1_weights, _ = get_gd_state_recorder_callback()
    l2_callback, l2_weights, _ = get_gd_state_recorder_callback()

    gamma = .95
    curr_eta = ExponentialLR(eta, gamma)

    l1_solver, l2_solver = GradientDescent(learning_rate=curr_eta, callback=l1_callback, \
        out_type=out_type),  GradientDescent(learning_rate=curr_eta, callback=l2_callback, \
        out_type=out_type)
    l1_solver.fit(f=L1(init), X=np.empty, y=np.empty)
    l2_solver.fit(f=L2(init), X=np.empty, y=np.empty)

    plot_descent_path(module=L1, descent_path=np.array(l1_weights),
                      title="L1-Norm Gradient Descent Path").show()
    plot_descent_path(module=L2, descent_path=np.array(l2_weights),
                      title="L2-Norm Gradient Descent Path").show()

def load_data(path: str = "C:/Users/fserv/Downloads/Year 4/Semester B/IML/IML-Projects/datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_test, y_test = X_test.to_numpy(), y_test.to_numpy()
     
    # Plotting convergence rate of logistic regression 
    # over SA heart disease data
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X=X_train, y=y_train)
    predicted_y_prob = logistic_regression.predict_proba(X=X_test)
    fpr, tpr, thresholds = roc_curve(y_test, predicted_y_prob)

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", 
        line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', 
              text=thresholds, name="", showlegend=False, marker_size=4,
                         marker_color="blue",
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))
    ).show()

    # Best threshold of logistic regression and its respective test error
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    logistic_regression = LogisticRegression(alpha=best_thresh)
    logistic_regression.fit(X=X_train, y=y_train)
    test_error = logistic_regression.loss(X=X_test, y=y_test)
    print("Best threshold: %.3f\n leads to test error of logistic regression: %.3f\n" % \
        (best_thresh, test_error))

    # Fitting regularized L1-2 logistic regression models, 
    # using cross-validation to specify values of regularization parameter
    lambdas = [.001, .002, .005, .01, .02, .05, .1]
    for penalty in ["l1", "l2"]:
        validation_errs = np.zeros(len(lambdas), dtype=float)
        for ix in range(len(lambdas)):
            gradient_descent = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
            logistic_regression = LogisticRegression(solver=gradient_descent, penalty=penalty, lam=lambdas[ix])
            validation_errs[ix] = cross_validate(logistic_regression, X_train, y_train, mean_squared_error)[1]

        # Lambda value achieving the best (min) validation error
        opt_lam = lambdas[np.argmin(validation_errs)]

        # Fitting logistic regression model with the best lambda
        logistic_regression = LogisticRegression(solver=gradient_descent, penalty=penalty, lam=opt_lam)
        logistic_regression.fit(X=X_train, y=y_train)
        test_error = logistic_regression.loss(X=X_test, y=y_test)
        print("Logistic regression with %s regulation term:\nBest lambda value: %.3f\n"
              "Test error with the best lambda: %.3f\n" % (penalty, opt_lam, test_error))

if __name__ == '__main__':
    np.random.seed(0)

    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
