"""Functions for computing thermodynamic quantities from the rotational
and vibrational partition functions of molecular hydrogen
"""

import numpy as np
from numba import vectorize, float64, boolean
from constants import BOLTZMANN, THETA_ROT, THETA_VIB, EPSILON

VECTORIZE_TARGET = "parallel"


@vectorize(
    [float64(float64, boolean)],
    fastmath=True,
    target=VECTORIZE_TARGET,
)
def erot_hydrogen(temp, ortho=True):
    """
    Average rotational energy of a molecular hydrogen molecule

    Parameters
    ----------
    temp: float or array_like
        Temperature in K
    ortho: boolean, optional
        True if you want ortho-H2, otherwise assumes para

    Returns
    -------
    Erot: float or array_like
        Mean vibrational energy at that temperature
    """
    error = 1e100
    z = 0.0
    dzdT = 0.0
    theta_beta = THETA_ROT / temp

    if ortho:
        j = 1
    else:
        j = 0

    while error > EPSILON:
        twojplusone = 2 * j + 1
        jjplusone = j * (j + 1)

        if ortho:
            expterm = 3 * np.exp(-theta_beta * (jjplusone - 2))
            zterm = twojplusone * expterm
            dzterm = twojplusone * expterm * (jjplusone - 2) * theta_beta / temp
        else:
            expterm = np.exp(-theta_beta * jjplusone)
            zterm = twojplusone * expterm
            dzterm = twojplusone * expterm * jjplusone * theta_beta / temp

        z += zterm
        dzdT += dzterm
        error = max(zterm / z, dzterm / dzdT)
        j += 2

    return BOLTZMANN * temp * temp * dzdT / z


@vectorize(
    [float64(float64)],
    fastmath=True,
    target=VECTORIZE_TARGET,
)
def evib_hydrogen(temp):
    """
    Mean vibrational energy of a hydrogen molecule

    Parameters
    ----------
    temp: float or array_like
        Temperature in K

    Returns
    -------
    Evib: float or array_like
        Mean vibrational energy at that temperature
    """
    return BOLTZMANN * THETA_VIB / np.expm1(THETA_VIB / temp)


def etot_hydrogen(temp, ortho_frac=0.75):
    """
    Mean total (trans+vib+rot) energy of a hydrogen molecule

    Parameters
    ----------
    temp: float or array_like
        Temperature in K
    ortho: boolean, optional
        True if you want ortho-H2, otherwise assumes para

    Returns
    -------
    Etot: float or array_like
        Mean total energy of hydrogen molecule at that temperature
    """
    erot = ortho_frac * erot_hydrogen(temp, True)
    erot += (1 - ortho_frac) * erot_hydrogen(temp, False)
    evib = evib_hydrogen(temp)
    return 1.5 * BOLTZMANN * temp + erot + evib
