""""
Definitions of various physical constantss in CGS units
"""

from astropy.constants import k_B, m_p
from astropy.units import eV
import numpy as np

BOLTZMANN = k_B.cgs.value  # in erg/K
THETA_ROT = 85.4  # in K
THETA_VIB = 6140  # in K
ELECTRONVOLT = (1 * eV).cgs.value
PROTONMASS = m_p.cgs.value
EPSILON = np.finfo(float).eps
