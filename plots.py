from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("Agg")
from partition_functions import molecular_hydrogen_energy
from constants import BOLTZMANN
import numpy as np


def do_plots():
    """Make plots of internal energy, heat capacity, and adiabatic index"""
    Tgrid = np.logspace(1, 4.5, 10**6)
    e, cv, gamma = molecular_hydrogen_energy(Tgrid)

    fig, ax = plt.subplots(1, 3, figsize=(8, 3))

    kw = {"color": "black"}
    ax[0].plot(Tgrid, e / (BOLTZMANN * Tgrid), **kw)
    ax[1].plot(Tgrid, cv / BOLTZMANN, **kw)
    ax[2].plot(Tgrid, gamma, **kw)

    for a in ax:
        a.set(xlabel=r"$T\,\left(\rm K\right)$", xscale="log")
    ax[0].set_ylabel(r"$\langle E \rangle/k_{\rm B} T$")
    ax[1].set_ylabel(r"$c_{\rm V} / k_{\rm B}$")
    ax[2].set_ylabel(r"$\Gamma_{\rm 1}$")
    fig.tight_layout()

    plt.savefig("H2_eos.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    do_plots()
