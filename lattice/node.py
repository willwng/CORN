"""
Class representing Vertices or Node in lattice
The position of each node does not change, it is only used to represent the
initial position

The instance variables should only be accessed via public methods below
since their internal implementation may change, but the methods behavior
will remain the same

Inheritance of object prevents formation of a __dict__ and reinforces use of __slots__
"""
from typing import Tuple


class Node(object):
    # Instance variables:
    # id : ID number of node
    # x, y : position (x, y) of node
    # boundary : Whether node is top or bottom
    # edge : Whether node is on the x edge (0 or lattice length)
    # has_bonds : Whether this node has any bonds attached to it
    # num_neighbors: The number of bonds attached to this node;
    # used for correlated networks

    # The use of slots instead of dicts should reduce memory usage
    __slots__ = ["id", "x", "y", "boundary", "edge", "top_edge", "hinge", "has_bonds", "num_neighbors"]
    id: int
    x: float
    y: float
    boundary: bool
    hinge: bool
    edge: bool
    top_edge: bool
    has_bonds: bool
    num_neighbors: int

    def __init__(self, x: float, y: float, n_id: int):
        self.x = x
        self.y = y
        self.boundary = False
        self.edge = False
        self.top_edge = False
        self.hinge = False
        self.id = n_id
        self.has_bonds = False
        self.num_neighbors = 0

    def add_bond(self) -> None:
        """
        :return: (void) mutates self with new node
        """
        self.has_bonds = True

    def is_bonded(self) -> bool:
        return self.has_bonds

    def get_xy(self) -> Tuple[float, float]:
        """
        :return: tuple containing (x,y) of node
        :rtype: tuple
        """
        return self.x, self.y

    def set_x(self, x: float) -> None:
        self.x = x

    def set_y(self, y: float) -> None:
        self.y = y

    def set_xy(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def get_id(self) -> int:
        return self.id

    def is_boundary(self) -> bool:
        return self.boundary

    def set_boundary(self) -> None:
        self.boundary = True

    def set_non_boundary(self) -> None:
        self.boundary = False

    def is_hinge(self) -> bool:
        return self.hinge

    def set_hinge(self):
        self.hinge = True

    def is_edge(self) -> bool:
        return self.edge

    def set_edge(self) -> None:
        self.edge = True

    def is_top_edge(self) -> bool:
        return self.top_edge

    def set_top_edge(self) -> None:
        self.top_edge = True

    # The following is for correlated network generation
    def add_neighbor(self) -> None:
        self.num_neighbors += 1

    def remove_neighbor(self) -> None:
        self.num_neighbors -= 1

    def get_num_neighbors(self) -> int:
        return self.num_neighbors

    def reset_neighbors(self) -> None:
        self.num_neighbors = 0
