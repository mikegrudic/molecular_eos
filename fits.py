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
    Tgrid = np.logspace(0, 4.5, 10**5)
    logT = np.log10(Tgrid)
    logzrot, e, cv, gamma = molecular_hydrogen_energy(Tgrid)
    z = np.exp(logzrot)

    def fitfunc(params):
        return (z.min() ** params[0] + (Tgrid / 10 ** params[1]) ** params[0]) ** (
            1.0 / params[0]
        )

    def fitfunc(params):
        return np.log(
            np.exp(z.min() ** params[0])
            + np.exp((Tgrid / 10 ** params[1]) ** params[0])
        ) ** (1.0 / params[0])

    def lossfunc(params):
        return np.sum(np.log(fitfunc(params) / z) ** 2)

    params = minimize(
        lossfunc,
        (1.0, 2.0),
    ).x
    print(params)

    plt.loglog(Tgrid, np.log(z), Tgrid, np.log(fitfunc(params)))
    plt.show()


if __name__ == "__main__":
    do_fits()
