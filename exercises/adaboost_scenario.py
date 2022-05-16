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
    adaboost_fitted_stump = AdaBoost(DecisionStump, n_learners)
    adaboost_fitted_stump.fit(train_X, train_y)

    train_error, test_error = np.zeros(n_learners), np.zeros(n_learners)

    for t in range(n_learners):
        train_error[t] = adaboost_fitted_stump.partial_loss(train_X, train_y, t + 1)
        test_error[t] = adaboost_fitted_stump.partial_loss(test_X, test_y, t + 1)

    plot = make_subplots(rows=1, cols=1)
    plot.add_trace(go.Scatter(x=np.arange(n_learners), y=train_error, mode='lines', marker_color="black", name="Train error"))
    plot.add_trace(go.Scatter(x=np.arange(n_learners), y=test_error, mode='lines', marker_color="yellow", name="Test error"))
    plot.update_layout(height=600, width=1000,
                              title_text=f"Training- and test- errors as function of #boosted fitted learners "
                                         "with noise = %.1f" % noise,
                              xaxis_title="#Fitted learners",
                              yaxis_title="Normalized Misclassification Error")
    plot.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    label_indicators = ((test_y + 1) * 0.5).astype(int)  # {-1,1} => {0,1}

    symbols = np.array(["circle", "x"])
    model_names = ["#classifiers = 5", "#classifiers = 50", "#classifiers = 100", "#classifiers = 250"]
    
    plot = make_subplots(rows=2, cols=2, subplot_titles=model_names)

    for ix, t in enumerate(T):
        plot.add_traces([decision_surface(lambda X: adaboost_fitted_stump.partial_predict(X, t), 
                                            lims[0], lims[1], showscale=False),
                           go.Scatter(x=test_X[:, 0], y=test_X[:, 1], 
                                mode="markers", showlegend=False,
                                marker=dict(color=label_indicators, symbol=symbols[label_indicators], colorscale=[custom[0], custom[-1]], \
                                line=dict(color="black", width=1)))], 
                           rows=(ix // 2) + 1, cols=(ix % 2) + 1)

    plot.update_layout(title_text="Boosted Decision Boundaries with noise = %.1f" % noise, title_x=.5,
            title_font_size=25, margin=dict(t=90)).update_xaxes(visible=False).update_yaxes(visible=False)
    plot.show()

    # Question 3: Decision surface of best performing ensemble
    plot = make_subplots(rows=1, cols=1)

    best_ensemble_size = np.argmin(test_error) + 1  # Index of the best ensemble + 1

    plot.add_traces([decision_surface(lambda X: adaboost_fitted_stump.partial_predict(X, best_ensemble_size), lims[0], lims[1], showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
    marker=dict(color=label_indicators, symbol=symbols[label_indicators], colorscale=[custom[0], custom[-1]]))], 
                    rows=1, cols=1)

    plot.update_layout(title_text="Decision surface of the best ensemble with noise = %.1f, ensemble size = "
                        "%d, Accuracy = %.4f" % (noise, best_ensemble_size, 1 - test_error[best_ensemble_size - 1]),
                    title_x=.5, title_font_size=25, margin=dict(t=90)).update_xaxes(visible=False).update_yaxes(visible=False)
    plot.show()

    # Question 4: Decision surface with weighted samples
    plot = make_subplots(rows=1, cols=1)

    sized_factor = 20
    sized_weights = (adaboost_fitted_stump.D_ / np.max(adaboost_fitted_stump.D_)) * sized_factor

    plot.add_traces([decision_surface(adaboost_fitted_stump.predict, lims[0], lims[1], showscale=False),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
            marker=dict(color=label_indicators, opacity=0.75, symbol=symbols[label_indicators], colorscale=[custom[0], custom[-1]], size=sized_weights))
            ], rows=1, cols=1)

    plot.update_layout(title_text="Decision surface using weighted samples and ensemble of size 250 and noise = %.1f)" % noise,
        title_x=.5, title_font_size=25, margin=dict(t=90)).update_xaxes(visible=False).update_yaxes(visible=False)
    plot.show()


if __name__ == '__main__':
    np.random.seed(0)

    noise_ratio = .0
    fit_and_evaluate_adaboost(noise_ratio)
    noise_ratio = .4
    fit_and_evaluate_adaboost(noise_ratio)
