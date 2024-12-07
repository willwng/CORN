"""
Types of protocols that can be run in the simulation.
    - BOND_REMOVAL: Remove bonds from the network until a certain threshold is reached
    - STRAIN_SWEEP: Apply a series of strains to the network
"""
from enum import Enum


class Protocol(Enum):
    BOND_REMOVAL = 0
    STRAIN_SWEEP = 1
