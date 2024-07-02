"""
Molecular EOS

Contains
--------
EOS - class implementing functions for relating internal energies, specific heats,
adiabatic index, pressure, etc for a partially-ionized hydrogen-helium mixture
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import root
from partition_functions import molecular_hydrogen_partition
from constants import BOLTZMANN, ELECTRONVOLT, PROTONMASS


@dataclass
class ChemicalState:
    """Container for the abundances of the various species that will specify
    the EOS"""

    f_mol: float = 1.0
    f_ortho: float = 0.75
    hydrogen_massfrac: float = 0.7381
    ionization_fraction: float = 0.0
    metallicity: float = 0.0134
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

    def internal_energy_permass(self, temp):
        """Returns internal energy per H as a function of temperature"""
        kT = BOLTZMANN * temp
        e_H2, _, _ = molecular_hydrogen_partition(temp)
        eps_H2 = 0.5 * self.cs.X * (1 - self.cs.y) * e_H2
        eps_HI = 1.5 * self.cs.X * (1 + self.cs.x) * self.cs.y * kT
        eps_He = 0.375 * self.cs.Y * kT
        # eps_diss = 4.48 * ELECTRONVOLT * self.cs.X * self.cs.y / 2
        return (eps_H2 + eps_HI + eps_He) / PROTONMASS  # + eps_diss

    def cV_permass(self, temp):
        """Returns heat capacity per H as a function of temperature"""
        _, cv, _ = molecular_hydrogen_partition(temp)
        cv_H2 = 0.5 * self.cs.X * (1 - self.cs.y) * cv
        cv_HI = 1.5 * self.cs.X * (1 + self.cs.x) * self.cs.y * BOLTZMANN
        cv_He = 0.375 * self.cs.Y * BOLTZMANN
        # eps_diss = 4.48 * ELECTRONVOLT * self.cs.X * self.cs.y / 2
        return (cv_H2 + cv_HI + cv_He) / PROTONMASS  # + eps_diss

    # def cV_permass(self, temp):
    #     return self.perH_to_permass(self.cV_perH(temp))

    def perH_to_permass(self, quantity):
        """Convert a per-unit-mass in CGS quantity to per-H-nucleus"""
        return quantity * (self.cs.X / PROTONMASS)

    def permass_to_perH(self, quantity):
        """Convert a per-H-nucleus quantity to per-unit-mass in CGS"""
        return quantity / (self.cs.X / PROTONMASS)

    def perH_to_perparticle(self, quantity):
        """Convert a per-H-nucleus quantity to per-unit-mass in CGS"""
        return quantity * self.mean_molecular_weight

    def internal_energy_perH(self, temp):
        """Returns internal energy per H"""
        u = self.internal_energy_permass(temp)
        return self.permass_to_perH(u)

    # def internal_energy_perparticle(self, temp):
    #     """Returns internal energy per particle"""
    #     return self.perH_to_perparticle(self.internal_energy_perH(temp))

    def pressure_from_temp(self, temp, density):
        """Returns pressure if temperature is provided"""
        n = density / (self.mean_molecular_weight * PROTONMASS)
        return n * BOLTZMANN * temp

    def pressure_from_u(self, u, density):
        """Returns pressure if temperature is provided"""
        temp = self.u_to_temp(u)
        n = density / (self.mean_molecular_weight * PROTONMASS)
        return n * BOLTZMANN * temp

    def u_to_temp(self, u, method="newton", n_iter=4):
        """Converts internal energy per unit mass to temperature"""

        def func(T):
            return self.internal_energy_permass(T) - u

        def jac(T):
            return self.cV_permass(T)

        Tguess = u * PROTONMASS / (1.5 * BOLTZMANN)

        if method == "newton":
            for _ in range(n_iter):
                dT = -func(Tguess) / jac(Tguess)  # newton iteration
                Tguess += dT
            return Tguess

        return root(func, Tguess).x

    def eos_gamma(self, temp):
        """Returns adiabatic index"""
        cv = np.gradient(self.internal_energy_perparticle(temp), temp)
        gamma = (cv / BOLTZMANN + 1) / (cv / BOLTZMANN)
        return gamma
