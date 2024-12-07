"""
Here is where we define all the lattice types that can be generated
"""
from enum import Enum


class LatticeType(Enum):
    KAGOME = 1
    TRIANGULAR = 2
    DOUBLE_TRIANGULAR = 3
    SQUARE = 4
