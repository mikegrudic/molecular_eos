""""
Definitions of various physical constantss in CGS units
"""

from astropy.constants import k_B
import numpy as np

BOLTZMANN = k_B.cgs.value  # in erg/K
THETA_ROT = 85.4  # in K
THETA_VIB = 5987  # in K
EPSILON = np.finfo(float).eps
