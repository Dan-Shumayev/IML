import matplotlib.pyplot as plt
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
    empirical_means = list()
    for i in range(10, 1001, 10):
        empirical_means.append(abs(UnivariateGaussian().fit(s[:i]).mu_ - 10))

    px.bar(x=range(10, 1001, 10), y=empirical_means, labels={'x':'Sample size', 'y':'Mean deviation'}, title="Mean deviation as a function of sample size").show()

    # Question 3 - Plotting Empirical PDF of fitted model
    # Following question: what's expected to see in the plot?
    # Answer: the Gaussian distribution density graph function (bell), that's centralized
    #           in the mean value (10), in the neighborhood of |lambda|=1, as the variance is 1.
    #               And, we indeed got such a graph.
    px.scatter(x=s, y=univar_gaussian_est.pdf(s), labels={'x':'Sample value', 'y':'PDF'}, title="PDF as a function of sample value").show()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    samples = np.random.multivariate_normal(np.array([0,0,4,0]), np.array([[1,0.2,0,0.5], [0.2,2,0,0], [0,0,1,0], [0.5,0,0,1]]), 1000)
    multi_est = MultivariateGaussian().fit(samples)

    print(f"Estimated expectation: {multi_est.mu_} \n\n Covariance matrix: {multi_est.cov_} \n\n")

    # Question 5 - Likelihood evaluation
    f1 = f3 = np.linspace(-10, 10, 200)
    f1_f3_pairs = np.transpose([np.tile(f1, len(f3)), np.repeat(f3, len(f1))])
    zeroed_coordinate = np.zeros(200 * 200)
    mu_s = np.vstack((f1_f3_pairs[:, 0], zeroed_coordinate, f1_f3_pairs[:, 1], zeroed_coordinate)).T
    ll_applied_on_mus = np.apply_along_axis(MultivariateGaussian.log_likelihood, 1, arr=mu_s, cov=np.array([[1,0.2,0,0.5], [0.2,2,0,0], [0,0,1,0], [0.5,0,0,1]]), X=samples)

    plt.imshow(ll_applied_on_mus.reshape(200,200), extent=[-10,10,-10,10])
    plt.title("log-likelihood as a function of a mean vector")
    plt.xlabel("f1")
    plt.ylabel("f3")
    plt.show()

    # Question 6 - Maximum likelihood
    # Answer: -0.050; 3.969

if __name__ == "__main__":
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
