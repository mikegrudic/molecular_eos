"""
Molecular EOS

Contains
--------
EOS - class implementing functions for relating internal energies, specific heats,
adiabatic index, pressure, etc for a partially-ionized hydrogen-helium mixture
"""

from dataclasses import dataclass


@dataclass
class ChemicalState:
    """Container for the abundances of the various species that will specify
    the EOS"""

    f_mol: float = 1.0
    f_ortho: float = 0.75
    hydrogen_massfrac: float = 0.76
    ionization_fraction: float = 0.0


class EOS:
    """Implements methods for computing thermodynamic quantities"""

    def __init__(self, state=ChemicalState()):
        self.chemical_state = state

    @property
    def chemical_state(self):
        """Chemical state that specifies the EOS"""
        return self._chemical_state

    @chemical_state.setter
    def chemical_state(self, state):
        self._chemical_state = state
