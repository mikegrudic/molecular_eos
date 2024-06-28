"""
Molecular EOS

Contains
--------
EOS - class implementing functions for relating internal energies, specific heats,
adiabatic index, pressure, etc for a partially-ionized hydrogen-helium mixture
"""

import numpy as np
from dataclasses import dataclass
from partition_functions import etot_molecular_hydrogen
from constants import BOLTZMANN, ELECTRONVOLT, PROTONMASS


@dataclass
class ChemicalState:
    """Container for the abundances of the various species that will specify
    the EOS"""

    f_mol: float = 1.0
    f_ortho: float = 0.75
    hydrogen_massfrac: float = 0.71
    ionization_fraction: float = 0.0
    metallicity: float = 0.014
    ortho_fraction: float = 0.75

    @property
    def helium_massfrac(self) -> float:
        """Returns helium mass fraction"""
        return 1.0 - self.hydrogen_massfrac - self.metallicity

    @property
    def X(self) -> float:
        """Hydrogen mass fraction"""
        return self.hydrogen_massfrac

    @property
    def Y(self) -> float:
        """Helium mass fraction"""
        return self.helium_massfrac

    @property
    def x(self) -> float:
        """rho(HII)/(rho(HII) + rho(HI))"""
        return self.ionization_fraction

    @property
    def y(self) -> float:
        """rho(HI)/(rho(HI) + rho(H2))"""
        return 1 - self.f_mol


class EOS:
    """Implements methods for computing thermodynamic properties of a H-H2-He-e mixture"""

    def __init__(self, state=ChemicalState()):
        self.chemical_state = state

    @property
    def chemical_state(self):
        """Chemical state that specifies the EOS"""
        return self._chemical_state

    @chemical_state.setter
    def chemical_state(self, state):
        self._chemical_state = state

    @property
    def cs(self):
        """Alias for chemical state"""
        return self._chemical_state

    @property
    def mean_molecular_weight(self):
        """Mean molecular weight"""
        Y = self.cs.Y
        X = self.cs.X
        y = self.cs.y
        x = self.cs.x
        return 4.0 / (2 * X * (1 + y + 2 * x * y) + Y)

    def internal_energy_perH(self, temp):
        """Returns internal energy per H as a function of temperature"""
        kT = BOLTZMANN * temp
        e_H2 = etot_molecular_hydrogen(temp, self.cs.ortho_fraction)
        eps_H2 = 0.5 * self.cs.X * (1 - self.cs.y) * e_H2
        eps_HI = 1.5 * self.cs.X * (1 + self.cs.x) * self.cs.y * kT
        eps_He = 0.375 * self.cs.Y * kT
        # eps_diss = 4.48 * ELECTRONVOLT * self.cs.X * self.cs.y / 2
        return eps_H2 + eps_HI + eps_He

    def internal_energy_permass(self, temp):
        """Returns internal energy per unit mass"""
        e_perH = self.internal_energy_perH(temp)
        return e_perH / PROTONMASS

    def internal_energy_perparticle(self, temp):
        """Returns internal energy per particle"""
        return self.internal_energy_perH(temp) * self.mean_molecular_weight

    def pressure_from_temp(self, temp, density):
        """Returns pressure if temperature is provided"""
        n = density / (self.mean_molecular_weight * PROTONMASS)
        return n * BOLTZMANN * temp

    def pressure_from_u(self, u, density):
        """Returns pressure if temperature is provided"""
        temp = self.u_to_temp(u)
        n = density / (self.mean_molecular_weight * PROTONMASS)
        return n * BOLTZMANN * temp

    def u_to_temp(self, u):
        """Converts internal energy per unit mass to temperature"""

        # rootfind internal_energy_permass = u for T
        return 0

    def eos_gamma(self, temp):
        """Returns adiabatic index"""
        cv = np.gradient(self.internal_energy_perparticle(temp), temp)
        gamma = (cv / BOLTZMANN + 1) / (cv / BOLTZMANN)
        return gamma
