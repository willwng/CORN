"""
Class representing Bond or Edge in lattice

The instance variables should only be accessed via public methods below 
"""
from lattice.node import Node
from typing import Tuple

import numpy as np
import math


class Bond(object):
    # Instance variables:
    # n_1, n_2   The two nodes that this edge bonds
    # present    Whether the bond is active
    # hor_pbc    Whether the bond is a horizontal periodic boundary condition
    # top_pbc    Whether the bond is a top periodic boundary condition
    # corner     We need this for polarized triangular lattices

    # The use of slots should save on memory usage compared to a normal class object
    __slots__ = ["n_1", "n_2", "present", "hor_pbc", "top_pbc", "corner", "direction", "s_key"]
    n_1: Node
    n_2: Node
    present: bool
    hor_pbc: bool
    top_pbc: bool
    corner: bool
    direction: int
    s_key: float

    def __init__(self, n_1: Node, n_2: Node, is_present: bool):
        """
        Constructor for each Bond
        :param n_1: First node (typically higher id)
        :param n_2: Second node
        :param is_present: Whether this bond exists
        """
        self.n_1 = n_1
        self.n_2 = n_2
        assert self.n_1 != self.n_2
        self.present = is_present
        self.hor_pbc = False
        self.top_pbc = False
        self.corner = False
        self.direction = -1
        self.s_key = 0

    def set_inactive(self) -> None:
        """
        Sets the existence of this bond to false, does not remove the object
        """
        self.present = False

    def set_active(self) -> None:
        """
        Sets the existence of this bond to true
        """
        self.present = True

    def get_xy_xy(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Returns a tuple consisting of [(x1, y1), (x2, y2)] where (x1, y1) is
        the position returned from running node1.get_xy()
        """
        return self.n_1.get_xy(), self.n_2.get_xy()

    def exists(self) -> bool:
        """
        Returns whether this bond exists
        """
        return self.present

    def is_hor_pbc(self) -> bool:
        """
        Returns whether this bond is a periodic boundary condition in the horizontal direction
        """
        return self.hor_pbc

    def is_top_pbc(self) -> bool:
        """
        Returns whether this bond is a top-down periodic boundary condition
        """
        return self.top_pbc

    def is_corner(self) -> bool:
        return self.corner

    def set_corner(self) -> None:
        self.corner = True

    def set_hor_pbc(self) -> None:
        """
        Sets this bond to be a periodic boundary condition
        """
        self.hor_pbc = True

    def set_top_pbc(self) -> None:
        """
        Sets this bond to be a top-down periodic boundary condition
        """
        self.top_pbc = True

    def get_node1(self) -> Node:
        """
        Returns the first node
        """
        assert self.n_1 != self.n_2
        return self.n_1

    def get_node2(self) -> Node:
        """
        Returns the second node
        """
        assert self.n_1 != self.n_2
        return self.n_2

    def get_other_node(self, node) -> Node:
        """
        Returns the other node that is not [node]
        """
        assert self.n_1 != self.n_2
        if self.n_1 == node:
            return self.n_2
        else:
            return self.n_1

    def get_direction(self) -> int:
        """
        Returns the direction of this bond
        0: horizontal, 1: right, 2: left

        """
        # If direction is not set, calculate it
        if self.direction == -1:
            pos1, pos2 = self.get_xy_xy()
            # Horizontal leaning bonds
            if pos1[1] == pos2[1]:
                self.direction = 0
            # Either left/right leaning bonds
            else:
                r_vector = np.subtract(pos2, pos1)
                angle = math.atan2(r_vector[1], r_vector[0]) % math.pi
                if angle < math.pi / 2:
                    self.direction = 1 if not self.is_hor_pbc() else 2
                else:
                    self.direction = 2 if not self.is_hor_pbc() else 1

        return self.direction

    def get_higher_pos(self) -> Tuple[float, float]:
        """
        Returns the position of the higher node

        """
        pos1, pos2 = self.get_xy_xy()
        return pos1 if pos1[1] > pos2[1] else pos2

    def get_skey(self) -> float:
        return self.s_key

    def set_skey(self, s_key: float) -> None:
        self.s_key = s_key
