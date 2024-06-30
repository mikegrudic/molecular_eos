"""Functions for computing thermodynamic quantities from the rotational
and vibrational partition functions of molecular hydrogen
"""

import numpy as np
from numba import njit, prange
from constants import BOLTZMANN, THETA_ROT, THETA_VIB, EPSILON

VECTORIZE_TARGET = "parallel"


@njit(parallel=True, fastmath=True)
def molecular_hydrogen_zrot(temp, ortho=True):
    """
    Partition function of hydrogen molecule and its derivatives

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
        expterm = 1.0
        dexpterm = 1.0
        if ortho:
            j = 1
            z = zterm = 9.0
            expterm = expmx * expmx * expmx8
        else:
            j = 0
            z = zterm = 1.0
            expterm = expmx8 / (expmx * expmx)

        while error > EPSILON:
            jjplusone = (j + 2) * ((j + 2) + 1)
            zterm *= (2 * j + 5) / (2 * j + 1) * expterm

            if ortho:
                # expterm = np.exp(-x * (jjplusone - 2))
                # zterm = 3 * twojplusone * expterm
                dzterm = (jjplusone - 2) * x / temp_i * zterm
                d2zterm = ((jjplusone - 2) * x - 2) * dzterm / temp_i
            else:
                # expterm = np.exp(-x * jjplusone)
                # zterm = twojplusone * expterm
                dzterm = jjplusone * x * zterm / temp_i
                d2zterm = (jjplusone * x - 2) * dzterm / temp_i

            z += zterm
            dz_dtemp += dzterm
            d2z_dtemp2 += d2zterm
            error = zterm / z
            expterm *= expmx8
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
def molecular_hydrogen_zrot2(temp, ortho=True):
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
    result = np.empty(N)

    for i in prange(N):
        error = 1e100
        z = 0.0
        temp_i = temp[i]
        # calculate everything in terms of dimmensionless x, convert to beta or T later
        x = THETA_ROT / temp_i
        expmx = np.exp(-x)
        expmx8 = np.power(expmx, 8)
        if ortho:
            j = 1
            zterm = 9.0
            expterm = expmx * expmx * expmx8
            z = zterm
        else:
            j = 0
            zterm = 1.0
            z = zterm
            expterm = expmx8 / (expmx * expmx)

        while error > EPSILON:
            zterm *= (2 * j + 5) / (2 * j + 1) * expterm
            z += zterm
            error = zterm / z
            expterm *= expmx8
            j += 2

        result[i] = z

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
