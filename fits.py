from partition_functions import molecular_hydrogen_energy
from constants import BOLTZMANN
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt


def sigmoid_sqrt(x):
    """cheap sigmoid function"""
    return 0.5 * (1 + x / np.sqrt(1 + x * x))


def do_fits():
    """Generate parameters for fits to thermodynamic quantities"""
    Tgrid = np.logspace(0, 4.5, 10**6)
    logT = np.log10(Tgrid)
    e, cv, gamma = molecular_hydrogen_energy(Tgrid)

    # gamma fit

    def fitfunc(params):
        return (
            5.0 / 3
            - params[0] * sigmoid_sqrt(params[1] * (logT - params[2]))
            - params[3] * sigmoid_sqrt(params[4] * (logT - params[5]))
        )

    def lossfunc(params):
        return np.sum(np.abs(fitfunc(params) - gamma) ** 2)

    params = minimize(lossfunc, np.random.rand(6)).x

    plt.loglog(Tgrid, gamma, Tgrid, fitfunc(params))
    plt.show()


if __name__ == "__main__":
    do_fits()
