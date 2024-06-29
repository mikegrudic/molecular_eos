"""Functions for computing thermodynamic quantities from the rotational
and vibrational partition functions of molecular hydrogen
"""

import numpy as np
from numba import vectorize, float64, boolean, njit, prange
from constants import BOLTZMANN, THETA_ROT, THETA_VIB, EPSILON

VECTORIZE_TARGET = "parallel"


@njit(parallel=True, fastmath=True)
def molecular_hydrogen_zrot(temp, ortho=True):
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

    if isinstance(temp, float):
        N = 1
        temp = np.array([temp])
    else:
        N = temp.shape[0]
    result = np.empty((N, 3))

    for i in prange(N):
        error = 1e100
        z = 0.0
        dz_dtemp = 0.0
        d2z_dtemp2 = 0.0
        temp_i = temp[i]
        theta_beta = THETA_ROT / temp_i

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
                dzterm = (jjplusone - 2) * theta_beta / temp_i * zterm
                d2zterm = (
                    ((jjplusone - 2) * THETA_ROT - 2 * temp_i)
                    * dzterm
                    / (temp_i * temp_i)
                )
            else:
                expterm = np.exp(-theta_beta * jjplusone)
                zterm = twojplusone * expterm
                # twojplusone * expterm * jjplusone * theta_beta / temp_i
                dzterm = jjplusone * theta_beta * zterm / temp_i
                d2zterm = (
                    (jjplusone * THETA_ROT - 2 * temp_i) * dzterm / (temp_i * temp_i)
                )

            z += zterm
            dz_dtemp += dzterm
            d2z_dtemp2 += d2zterm
            # print(zterm / z, dzterm / dz_dtemp, d2zterm / d2z_dtemp2)
            error = zterm / z  # max(zterm / z, dzterm / dz_dtemp, d2zterm / d2z_dtemp2)
            j += 2

        result[i, 0] = z
        result[i, 1] = BOLTZMANN * temp_i * temp_i * dz_dtemp / z
        result[i, 2] = (
            BOLTZMANN
            * temp_i
            * (2 * dz_dtemp + temp_i * d2z_dtemp2 - temp_i * dz_dtemp * dz_dtemp / z)
            / z
        )

    return result


@njit(parallel=True, fastmath=True)
def molecular_hydrogen_zvib(temp):
    """
    Mean vibrational energy of a hydrogen molecule

    Parameters
    ----------
    temp: float or array_like
        Temperature in K

    Returns
    -------
    result: ndarray
        Shape (N,3) array where each row contains 1. the partition function value,
        2. the average energy per molecule, and 3. the constant-volume heat capacity
    """

    if isinstance(temp, float):
        N = 1
        temp = np.array([temp])
    else:
        N = temp.shape[0]
    result = np.empty((N, 3))

    for i in prange(N):
        result[i, 0] = -1.0 / np.expm1(-THETA_VIB / temp[i])
        result[i, 1] = BOLTZMANN * THETA_VIB / np.expm1(THETA_VIB / temp[i])
        result[i, 2] = THETA_VIB * result[i, 0] * result[i, 1] / (temp[i] * temp[i])
    return result


def molecular_hydrogen_energy(temp, ortho_frac=0.75):
    """
    Partition function properties of a mixture of para- and ortho-hydrogen

    Parameters
    ----------
    temp: float or array_like
        Temperature in K
    ortho_frac: float, optional
        Fraction of ortho-H2 (default is 3:1 ortho:para mixture)

    Returns
    -------
    etot, cv, gamma: tuple
        tuple of ndarrays containing the average energy and constant volume heat
        capacity per molecule in CGS, and adiabatic index
    """

    para_frac = 1 - ortho_frac
    zrot_ortho = molecular_hydrogen_zrot(temp, True)
    zrot_para = molecular_hydrogen_zrot(temp, False)
    zvib = molecular_hydrogen_zvib(temp)
    etot = 1.5 * BOLTZMANN * temp  # translation
    cv = 1.5 * BOLTZMANN
    etot += ortho_frac * zrot_ortho[:, 1]  # ortho rotation
    cv += ortho_frac * zrot_ortho[:, 2]
    etot += para_frac * zrot_para[:, 1]  # para rotation
    cv += para_frac * zrot_para[:, 2]
    etot += zvib[:, 1]  # vibration
    cv += zvib[:, 2]
    gamma = (cv / BOLTZMANN + 1) / (cv / BOLTZMANN)
    zrot_total = ortho_frac * np.log(zrot_ortho[:, 0]) + para_frac * np.log(
        zrot_para[:, 0]
    )
    return zrot_total, etot, cv, gamma
