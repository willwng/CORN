"""
Numerical minimization methods for relaxing the network energy

All methods beginning with "LINEAR" solve the linearized energy (leading order in displacement)
- LINEAR: Directly solve using conjugate gradients

Nonlinear methods solve the full energy forms
- NONLINEAR: Solve the full energy using some form of nonlinear conjugate gradients
"""
from enum import Enum


class MinimizationType(Enum):
    LINEAR = 0
    TRUST_NEWTON_CG = 1
    TRUST_CONSTR = 2
    FIRE = 3
    FIRE2 = 4 # Fire2 is based on https://doi.org/10.1016/j.commatsci.2020.109584
