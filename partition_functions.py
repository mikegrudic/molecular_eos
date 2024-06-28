"""Functions for computing thermodynamic quantities from the rotational
and vibrational partition functions of molecular hydrogen
"""

import numpy as np
from numba import vectorize, float64, boolean, njit
from constants import BOLTZMANN, THETA_ROT, THETA_VIB, EPSILON

VECTORIZE_TARGET = "parallel"


@njit
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
    result: ndarray
        Shape (N,3) array containing the rotational partition function value,
        each row returns the average rotational energy per molecule, and the
        heat capacity per molecule at constant volume
    """
    error = 1e100
    z = 0.0
    dz_dtemp = 0.0
    d2z_dtemp2 = 0.0
    theta_beta = THETA_ROT / temp

    result = np.empty(3)

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
            dzterm = (jjplusone - 2) * theta_beta / temp * zterm
            d2zterm = ((jjplusone - 2) * THETA_ROT - 2 * temp) * dzterm / (temp * temp)
        else:
            expterm = np.exp(-theta_beta * jjplusone)
            zterm = twojplusone * expterm
            dzterm = (
                jjplusone * theta_beta * zterm / temp
            )  # twojplusone * expterm * jjplusone * theta_beta / temp
            d2zterm = (jjplusone * THETA_ROT - 2 * temp) * dzterm / (temp * temp)

        z += zterm
        dz_dtemp += dzterm
        d2z_dtemp2 += d2zterm
        error = max(zterm / z, dzterm / dz_dtemp, d2zterm / d2z_dtemp2)
        j += 2

    result[0] = z
    result[1] = BOLTZMANN * temp * temp * dz_dtemp / z
    result[2] = BOLTZMANN * temp * (2 * dz_dtemp + temp * d2z_dtemp2) / z
    return result


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


def etot_molecular_hydrogen(temp, ortho_frac=0.75):
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
