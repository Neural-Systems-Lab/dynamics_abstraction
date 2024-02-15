import numpy as np
import pdb
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


def gauss2d(mu, sigma, to_plot=False):
    w, h = 50, 50

    std = [np.sqrt(sigma[0]), np.sqrt(sigma[1])]
    x = np.linspace(mu[0] - 0.5 * std[0], mu[0] + 0.5 * std[0], w)
    y = np.linspace(mu[1] - 0.5 * std[1], mu[1] + 0.5 * std[1], h)

    x, y = np.meshgrid(x, y)

    x_ = x.flatten()
    y_ = y.flatten()
    xy = np.vstack((x_, y_)).T

    normal_rv = multivariate_normal(mu, sigma)
    z = normal_rv.pdf(xy)
    z = z.reshape(w, h, order='F')

    if to_plot:
        plt.contourf(x, y, z.T, levels=1, alpha=0.5)
        # plt.contourf(x, y, z.T)
        # plt.savefig("gauss2d.png", dpi=400)
        # plt.show()
        

    return z


MU = [50, 70]
SIGMA = [75.0, 90.0]
z = gauss2d(MU, SIGMA, True)