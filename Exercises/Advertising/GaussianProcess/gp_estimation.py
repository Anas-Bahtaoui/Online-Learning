import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# For some reason this is not working as expected :(


def n(x):
    return (1 - np.exp(-5 * x)) * 100


def generate_observations(x, noise_std):
    return n(x) + np.random.normal(0, noise_std, size=n(x).shape)


n_obs = 50
bids = np.linspace(0, 1, 20)
x_obs = np.array([])
y_obs = np.array([])

noise_std = 5

# We don't need to normalize the data because the values are already in the
# range [0, 1]
for i in range(n_obs):
    x_obs = np.append(x_obs, np.random.choice(bids, 1))
    y_obs = np.append(y_obs, generate_observations(x_obs[-1], noise_std))
    X = np.atleast_2d(x_obs).T
    y = y_obs.ravel()
    # Kernel function
    theta = 1.0
    l = 1.0
    kernel = C(theta, (1e-3, 1e3)) * RBF(l, (1e-3, 1e3))
    # Hyperparameters

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=noise_std ** 2, normalize_y=True)
    # Fit
    gp.fit(X, y)
    # Estimate hyper parameters

    x_pred = np.atleast_2d(bids).T
    y_pred, sigma = gp.predict(x_pred, return_std=True)

    plt.figure(i)
    plt.plot(x_pred, n(x_pred), 'r:', label=r'$n(x)$')
    plt.plot(X.ravel(), y, 'ro', label=r'Observed clicks')
    plt.plot(x_pred, y_pred, 'b-', label=r'Predicted clicks')
    plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
             np.concatenate([y_pred - 1.96 * sigma,
                             (y_pred + 1.96 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$n(x)$')
    plt.legend(loc='lower right')
    plt.show()
