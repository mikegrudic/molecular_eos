from partition_functions import molecular_hydrogen_partition
from astropy.constants import k_B as BOLTZMANN
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import sympy as sp

BOLTZMANN = BOLTZMANN.cgs.value


def sigmoid_sqrt(x):
    """cheap sigmoid function"""
    return 0.5 * (1 + x / np.sqrt(1 + x * x))


def sigmoid_exp(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_tanh(x):
    return 0.5 * (1 + np.tanh(x))


# return 1 / (1 + np.exp(-x))


# sigmoid = sigmoid_sqrt


def do_fits():
    """Generate parameters for fits to thermodynamic quantities"""
    Tgrid = np.logspace(1, 5, 10**5)
    logT = np.log10(Tgrid)
    etot, cv, _ = molecular_hydrogen_partition(Tgrid)
    print(etot.dtype)

    etot_beta = etot / BOLTZMANN / Tgrid
    cv = cv/BOLTZMANN

    sigmoid = sigmoid_exp

    def fitfunc(params):
        return (
            1.5 + sigmoid(np.polyval(params[::2], np.log10(Tgrid))) + sigmoid(np.polyval(params[1::2], np.log10(Tgrid)))
        )

    def lossfunc(params):
        return np.sum(
            np.abs(fitfunc(params) - etot_beta) ** 2
        )  # np.sum(np.abs(np.log(fitfunc(params) / etot_beta)) ** 2)

    plt.loglog(Tgrid, etot_beta, color="black")
    params = np.ones(2)
    errors = []
    fitparams = []
    # find the optimal polynomial order to use as argument to the fitting function
    for i in range(5):
        params = minimize(
            lossfunc,
            list(params) + [0.0, 0.0],
        ).x
        fitparams.append(params)
        errors.append(np.std(etot_beta - fitfunc(params)) ** 2)
    params = fitparams[np.array(errors).argmin()]

    logT = sp.log(sp.Symbol("T"), 10.0)

    def sympy_polyval(params, x):
        N = len(params)
        return sp.horner(sum([c * x ** (N - i - 1) for i, c in enumerate(params)]))

    p1 = sympy_polyval(params[::2], logT)
    p2 = sympy_polyval(params[1::2], logT)

    def sigmoid_symbolic(x):
        return 1 / (1 + sp.exp(-x))

    #    def sigmoid_symbolic(x):
    #        return 0.5 * (1 + sp.tanh(x))

    symbolic_fit = 1.5 + sigmoid_symbolic(p1) + sigmoid_symbolic(p2)
    func = sp.lambdify(sp.Symbol("T"), symbolic_fit)

    with open("energy_fit.py", "w") as F:
        code = f"""
import sympy as sp
def H2_energy_over_kb(T):
    p1 = {str(p1).replace("exp", "sp.exp").replace("log", "sp.log")}
    p2 = {str(p2).replace("exp", "sp.exp").replace("log", "sp.log")}
    return 1.5 + 1/(1+sp.exp(-p1)) + 1/(1+sp.exp(-p2))
"""
        F.write(code)

    from energy_fit import H2_energy_over_kb

    T = sp.Symbol("T")
    numerical_func = sp.lambdify(T, H2_energy_over_kb(T))
    assert np.all(np.isclose(numerical_func(Tgrid), etot_beta, atol=0.01))
    np.save("T_vs_ebeta_cv.npy", np.c_[Tgrid, etot_beta, cv])

if __name__ == "__main__":
    do_fits()
