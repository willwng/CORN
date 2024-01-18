"""
    This file helps set the activity of bonds in a lattice (used for oriented networks).
    The "basin" protocol allows for a value of `p` to directly translate to whether a bond is active or not (consider
        each bond as a "basin" and the value of `p` as the water level)
"""
from collections import deque
from typing import Deque

from numpy.random import Generator

from lattice.abstract_lattice import AbstractLattice
from lattice.bond import Bond


def assign_bond_levels(lattice: AbstractLattice, rng: Generator):
    """ Sets the bond levels (s_i):  the value for when p > s_i, the bond i is added to the lattice """
    # Sort the bonds so we can get reproducible results
    sorted_bonds = sorted(lattice.get_bonds(), key=lambda b: b.get_node1().get_id())
    for bond in sorted_bonds:
        bond.set_skey(rng.uniform())
    return


def set_bonds_basin(lattice: AbstractLattice, p: float, r: float, target_direction: int):
    """ Sets the bonds given p and r """
    threshold_target = (3 * r * p) / (2 + r)
    threshold_other = (3 * p) / (2 + r)
    for bond in lattice.get_bonds():
        # if bond.is_boundary():
        #     continue
        # Threshold based on direction
        threshold = threshold_target if bond.get_direction() == target_direction else threshold_other
        if bond.get_skey() < threshold:
            bond.add_bond()
        else:
            bond.remove_bond()
    lattice.update_active_bonds()
    return


def get_removal_order(lattice: AbstractLattice, r: float, target_direction: int) -> Deque[Bond]:
    """
    Returns the bonds in the order that they should be removed. Ignoring bonds that are already removed.
    To be called after [assign_bond_levels]
    """

    def get_add_p(bond: Bond, r_s: float) -> float:
        """Returns the probability at which the bond should be added"""
        s_i = bond.get_skey()
        if bond.get_direction() == target_direction:
            p_add = (2 + r_s) * s_i / (3 * r_s)
        else:
            p_add = (2 + r_s) * s_i / 3
        return p_add

    # Sort in the order that they should be removed, then filter out the ones that are already removed
    removal_order = list(sorted(lattice.get_bonds(), key=lambda b: get_add_p(b, r), reverse=True))
    removal_order = list(filter(lambda b: b.exists() and not b.is_boundary(), removal_order))
    return deque(removal_order)
