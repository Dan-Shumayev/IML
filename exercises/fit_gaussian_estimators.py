import sys

sys.path.append("C:\\Users\\fserv\\Downloads\\Year 4\\Semester B\\IML\\IML-Projects")
# TODO - remove sys and resolve Module not found issue

import numpy as np
import plotly.express as px
import plotly.io as pio
from IMLearn.learners import MultivariateGaussian, UnivariateGaussian

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    s = np.random.normal(10, 1, 1000)
    univar_gaussian_est = UnivariateGaussian().fit(s)

    print(f"({univar_gaussian_est.mu_}, {univar_gaussian_est.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    ys = list()
    for i in range(10, 1001, 10):
        ys.append(abs(UnivariateGaussian().fit(s[:i]).mu_ - 10))

    px.bar(x=range(10, 1001, 10), y=ys, labels={'x':'Sample size', 'y':'Mean deviation'}, title="Mean deviation as a function of sample size").show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pass

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == "__main__":
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
