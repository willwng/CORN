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
    # id      The identifier of the node
    # x, y    The position of the node

    # The use of slots instead of dicts should reduce memory usage
    __slots__ = ["id", "x", "y"]

    def __init__(self, x: float, y: float, n_id: int):
        self.x = x
        self.y = y
        self.id = n_id

    def get_xy(self) -> Tuple[float, float]:
        """
        :return: tuple containing (x,y) of node
        :rtype: tuple
        """
        return self.x, self.y

    def set_xy(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def get_id(self) -> int:
        return self.id
