import numpy as np
from typing import Tuple


def get_transformation_matrices(gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the stretch x, stretch y, shear, dilate transformation matrices
    """
    stretch_x = np.array([[1 + gamma, 0], [0, 1]])
    stretch_y = np.array([[1, 0], [0, 1 + gamma]])
    shear = np.array([[1, gamma], [gamma, 1]])
    dilate = np.array([[1 + gamma, 0], [0, 1 + gamma]])
    return stretch_x, stretch_y, shear, dilate


def transform_pos_matrix(pos_matrix: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Returns the transformed position matrix
    """
    pos_matrix = pos_matrix.reshape((-1, 2))
    return (transformation_matrix @ pos_matrix.T).T
