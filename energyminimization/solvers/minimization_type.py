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
    NONLINEAR = 1
    FIRE = 2
