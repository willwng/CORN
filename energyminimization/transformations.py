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
    def __init__(self, gamma: float):
        self.gamma = gamma
        self.transformation = self.create_transformation()

    def create_transformation(self):
        raise NotImplementedError

    def update_gamma(self, gamma: float):
        self.gamma = gamma
        self.transformation = self.create_transformation()

    def apply(self, pos_matrix: np.ndarray) -> np.ndarray:
        return transform_pos_matrix(pos_matrix, self.transformation)


class StretchX(Strain):
    def __init__(self, gamma: float):
        super().__init__(gamma)
        self.name = "stretch_x"

    def create_transformation(self):
        return np.array([[1 + self.gamma, 0], [0, 1]])


class StretchY(Strain):
    def __init__(self, gamma: float):
        super().__init__(gamma)
        self.name = "stretch_y"

    def create_transformation(self):
        return np.array([[1, 0], [0, 1 + self.gamma]])


class Shear(Strain):
    def __init__(self, gamma: float):
        super().__init__(gamma)
        self.name = "shear"

    def create_transformation(self):
        return np.array([[1, self.gamma], [self.gamma, 1]])


class Dilate(Strain):
    def __init__(self, gamma: float):
        super().__init__(gamma)
        self.name = "dilate"

    def create_transformation(self):
        return np.array([[1 + self.gamma, 0], [0, 1 + self.gamma]])
