import sys  # TODO - remove!
from typing import Tuple

sys.path.append("c:\\Users\\fserv\\Downloads\\Year 4\\Semester B\\IML\\IML-Projects")  # TODO - remove!
import numpy as np
import plotly.graph_objects as go
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners import AdaBoost
from plotly.subplots import make_subplots
from utils import *


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    train_X, train_y = generate_data(train_size, noise_ratio)
    test_X, test_y = generate_data(test_size, noise_ratio)

    adaboost_fitted_stump = AdaBoost(DecisionStump, n_learners)
    adaboost_fitted_stump._fit(train_X, train_y)

    train_error = test_error = np.zeros(n_learners)

    for t in range(1, n_learners):
        train_error[t] = adaboost_fitted_stump.partial_loss(train_X, train_y, t)
        test_error[t] = adaboost_fitted_stump.partial_loss(test_X, test_y, t)

    plot = go.Figure()
    plot.add_trace(go.Scatter(x=np.arange(n_learners), y=train_error, name="Train error"))
    plot.add_trace(go.Scatter(x=np.arange(n_learners), y=test_error, name="Test error"))
    plot.show()

    # Question 2: Plotting decision surfaces
    # T = [5, 50, 100, 250]
    # lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    # model_names = ["Number of classifiers = 5", "Number of classifiers = 50", "Number of classifiers = 100",
    #                "Number of classifiers = 250"]
    # symbols = np.array(["circle", "x"])

    # fig_q2 = make_subplots(rows=2, cols=2, subplot_titles=model_names,
    #                     horizontal_spacing=0.01, vertical_spacing=.03)

    # # Convert labels from {-1,1} to {0,1}
    # labels = (0.5 * test_y + 0.5).astype(int)
    # for i, t in enumerate(T):

    #     def predict(X: np.ndarray) -> np.ndarray:
    #         return adaboost.partial_predict(X, t)
    #     fig_q2.add_traces([decision_surface(predict, lims[0], lims[1], showscale=False),
    #                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
    #                                   marker=dict(color=labels, symbol=symbols[labels],
    #                                               colorscale=[custom[0], custom[-1]], line=dict(color="black", width=1))
    #                                   )
    #                        ], rows=(i // 2) + 1, cols=(i % 2) + 1)

    # fig_q2.update_layout(title_text="Decision Boundaries Of ADABoot (Noise = %.1f)" % noise, title_x=0.5,
    #                      title_font_size=30, margin=dict(t=100)) \
    #     .update_xaxes(visible=False).update_yaxes(visible=False)

    # fig_q2.show()

    # Question 3: Decision surface of best performing ensemble
    # fig_q3 = make_subplots(rows=1, cols=1)
    # ensemble_size = iterations_num[np.argmin(test_errors)]

    # def predict(X: np.ndarray) -> np.ndarray:
    #     return adaboost.partial_predict(X, ensemble_size)

    # fig_q3.add_traces([decision_surface(predict, lims[0], lims[1], showscale=False),
    #                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
    #                               marker=dict(color=labels, symbol=symbols[labels],
    #                                           colorscale=[custom[0], custom[-1]])
    #                               )
    #                    ], rows=1, cols=1)
    # fig_q3.update_layout(title_text="Decision surface of best performing ensemble (Noise = %.1f): Ensemble size = "
    #                                 "%d, Accuracy = %.3f" %
    #                                 (noise, ensemble_size, 1 - test_errors[ensemble_size - 1]),
    #                      title_x=0.5, title_font_size=30, margin=dict(t=100)).\
    #     update_xaxes(visible=False).update_yaxes(visible=False)

    # fig_q3.show()

    # Question 4: Decision surface with weighted samples
    # fig_q4 = make_subplots(rows=1, cols=1)
    # weights = (adaboost.D_ / np.max(adaboost.D_)) * (25 if noise > 0 else 50)
    # # Convert labels from {-1,1} to {0,1}
    # labels = (0.5 * train_y + 0.5).astype(int)
    # fig_q4.add_traces([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
    #                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
    #                               marker=dict(color=labels, opacity=0.9, symbol=symbols[labels],
    #                                           colorscale=[custom[0], custom[-1]], size=weights)
    #                               )
    #                    ], rows=1, cols=1)
    # fig_q4.update_layout(title_text="Decision surface with weighted samples (Ensemble size = 250, Noise = %.1f)" % noise,
    #                      title_x=0.5, title_font_size=30, margin=dict(t=100)). \
    #     update_xaxes(visible=False).update_yaxes(visible=False)

    # fig_q4.show()


if __name__ == '__main__':
    np.random.seed(0)

    noise_ratio = .0
    fit_and_evaluate_adaboost(noise_ratio)
    noise_ratio = .4
    # fit_and_evaluate_adaboost(noise_ratio)
