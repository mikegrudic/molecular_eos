"""Functions for computing thermodynamic quantities from the rotational
and vibrational partition functions of molecular hydrogen
"""

import numpy as np
from numba import njit, prange
from constants import BOLTZMANN, THETA_ROT, THETA_VIB, EPSILON

VECTORIZE_TARGET = "parallel"


@njit(parallel=True, fastmath=True)
def molecular_hydrogen_zrot_mixture(temp, ortho_frac=0.75):
    """
    Rotational partition function of hydrogen molecule and derived quantities,
    considering a mixture of ortho- and parahydrogen that cannot efficiently
    come into equilibrium.

    Parameters
    ----------
    temp: float or array_like
        Temperature in K
    ortho_frac: float, optional
        Fraction of ortho-H2 (default is 3:1 ortho:para mixture)

    Returns
    -------
    result: ndarray
        Shape (N,3) array where each row stores the partition function value,
        the average rotational energy per molecule, and the heat capacity per
        molecule at constant volume.
    """

    if isinstance(temp, float):
        N = 1
        temp = np.array([temp])
    else:
        N = temp.shape[0]
    result = np.empty((N, 3))

    para_frac = 1 - ortho_frac

    for i in prange(N):
        error = 1e100
        z = np.zeros(2)  # 0 for para, 1 for ortho
        dz_dtemp = np.zeros(2)
        d2z_dtemp2 = np.zeros(2)
        zterm = np.zeros(2)

        x = THETA_ROT / temp[i]
        expmx = np.exp(-x)
        expmx4 = np.power(expmx, 4)
        z[1] = zterm[1] = 9.0
        z[0] = zterm[0] = 1.0
        expterm = expmx4 * expmx * expmx
        # print(expterm)
        # Summing over rotational levels
        j = 2
        while error > EPSILON:
            s = j % 2
            jjplusone = j * (j + 1)
            zterm[s] *= (2 * j + 1) / (2 * j - 3) * expterm
            if s == 1:  # ortho
                dzterm = (jjplusone - 2) * x * zterm[1]
                d2zterm = ((jjplusone - 2) * x - 2) * dzterm
            else:  # para
                dzterm = jjplusone * x * zterm[0]
                d2zterm = (jjplusone * x - 2) * dzterm
            z[s] += zterm[s]
            dz_dtemp[s] += dzterm
            d2z_dtemp2[s] += d2zterm
            error = max(zterm[0] / z[0], zterm[1] / z[1])
            expterm *= expmx4
            j += 1

        result[i, 0] = np.exp(para_frac * np.log(z[0]) + ortho_frac * np.log(z[1]))  # partition function
        result[i, 1] = (
            BOLTZMANN * temp[i] * (para_frac * dz_dtemp[0] / z[0] + ortho_frac * dz_dtemp[1] / z[1])
        )  # mean energy
        result[i, 2] = BOLTZMANN * (
            ortho_frac * (2 * dz_dtemp[1] + d2z_dtemp2[1] - dz_dtemp[1] * dz_dtemp[1] / z[1]) / z[1]
            + para_frac * (2 * dz_dtemp[0] + d2z_dtemp2[0] - dz_dtemp[0] * dz_dtemp[0] / z[0]) / z[0]
        )  # heat capacity

    return result


@njit(parallel=True, fastmath=True)
def molecular_hydrogen_zrot_simple(temp, ortho=True):
    """
    Partition function of hydrogen molecule and its derivatives for a single
    spin isomer

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
        x = THETA_ROT / temp_i
        expmx = np.exp(-x)
        expmx8 = np.power(expmx, 8)
        if ortho:
            j = 1
            z = zterm = 9.0
            expterm = expmx * expmx * expmx8
        else:
            j = 0
            z = zterm = 1.0
            expterm = expmx8 / (expmx * expmx)

        # Summing over rotational levels
        while error > EPSILON:
            j += 2
            jjplusone = j * (j + 1)
            zterm *= (2 * j + 1) / (2 * j - 3) * expterm

            if ortho:
                dzterm = (jjplusone - 2) * x * zterm
                d2zterm = ((jjplusone - 2) * x - 2) * dzterm
            else:
                dzterm = jjplusone * x * zterm
                d2zterm = (jjplusone * x - 2) * dzterm
            z += zterm
            dz_dtemp += dzterm
            d2z_dtemp2 += d2zterm
            error = zterm / z
            expterm *= expmx8

        result[i, 0] = z
        result[i, 1] = BOLTZMANN * temp_i * dz_dtemp / z
        result[i, 2] = BOLTZMANN * (2 * dz_dtemp + d2z_dtemp2 - dz_dtemp * dz_dtemp / z) / z

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


def molecular_hydrogen_partition(temp, ortho_frac=0.75, return_z=False):
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

    ztrans = temp**1.5
    zrot = molecular_hydrogen_zrot_mixture(temp, ortho_frac)
    zvib = molecular_hydrogen_zvib(temp)
    etot = 1.5 * BOLTZMANN * temp  # translation
    cv = 1.5 * BOLTZMANN
    etot += zrot[:, 1]  # rotation
    cv += zrot[:, 2]
    etot += zvib[:, 1]  # vibration
    cv += zvib[:, 2]
    gamma = (cv / BOLTZMANN + 1) / (cv / BOLTZMANN)
    if return_z:
        logz = np.log(zrot[:, 0]) + np.log(zvib[:, 0]) + np.log(ztrans)
        return logz, etot, cv, gamma
    return etot, cv, gamma
