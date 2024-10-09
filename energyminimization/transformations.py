"""
This file contains a class for representing strains, as well as classes for specific strains
"""
import numpy as np


def transform_pos_matrix(pos_matrix: np.ndarray, transformation_matrix) -> np.ndarray:
    """
    Returns the transformed position matrix
    """
    pos_matrix = pos_matrix.reshape((-1, 2))
    return (transformation_matrix @ pos_matrix.T).T


class Strain:
    name: str
    def __init__(self, gamma: float, transformation: np.ndarray):
        self.gamma = gamma
        self.transformation = transformation

    def apply(self, pos_matrix: np.ndarray) -> np.ndarray:
        return transform_pos_matrix(pos_matrix, self.transformation)


class StretchX(Strain):
    def __init__(self, gamma: float):
        transformation = np.array([[1 + gamma, 0], [0, 1]])
        super().__init__(gamma, transformation)
        self.name = "stretch_x"


class StretchY(Strain):
    def __init__(self, gamma: float):
        transformation = np.array([[1, 0], [0, 1 + gamma]])
        super().__init__(gamma, transformation)
        self.name = "stretch_y"


class Shear(Strain):
    def __init__(self, gamma: float):
        transformation = np.array([[1, gamma], [gamma, 1]])
        super().__init__(gamma, transformation)
        self.name = "shear"


class Dilate(Strain):
    def __init__(self, gamma: float):
        transformation = np.array([[1 + gamma, 0], [0, 1 + gamma]])
        super().__init__(gamma, transformation)
        self.name = "dilate"
