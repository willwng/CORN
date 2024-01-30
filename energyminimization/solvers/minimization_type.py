"""
Numerical minimization methods for relaxing the network energy

All methods beginning with "LINEAR" solve the linearized energy (leading order in displacement)
- LINEAR: Directly solve using conjugate gradients
- LINEAR_RELAX:
- LINEAR_FIRE: Solve using FIRE algorithm
- LINEAR_GPU: Solve using conjugate gradients on GPU
- LINEAR_PRE: Solve using preconditioned conjugate gradients
- LINEAR_PRE_GPU: Solve using preconditioned conjugate gradients on GPU

Nonlinear methods solve the full energy forms
- NONLINEAR: Solve the full energy using some form of nonlinear conjugate gradients
"""
from enum import Enum


class MinimizationType(Enum):
    LINEAR = 0
    LINEAR_RELAX = 1
    LINEAR_FIRE = 2
    LINEAR_GPU = 3
    LINEAR_PRE = 4
    LINEAR_PRE_GPU = 5
    NONLINEAR = 6
