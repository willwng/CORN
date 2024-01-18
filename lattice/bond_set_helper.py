"""
Class used to help set bonds
"""

from typing import Dict, Optional, Callable, List
from lattice.bond import Bond
import numpy as np
import random


class BondSetter:
    probabilities: Dict[Bond, float]  # Maps a bond to the probability that it exists
    get_prob: Callable[[Bond], float]  # Function from bond to probability
    updated_bonds: Callable[
        [Bond], List[Bond]
    ]  # Determines the bonds needing updated probabilities after setting one
    debug = set()

    def __init__(
        self,
        all_bonds: List[Bond],
        get_prob: Callable[[Bond], float],
        updated_bond: Callable[[Bond], List[Bond]],
    ):
        self.get_prob = get_prob
        self.updated_bonds = updated_bond
        # Create the map from bonds to probabilities (only include non-active bonds)
        probs: Dict[Bond, float] = {}
        for bond in all_bonds:
            if bond.exists():
                continue
            probs[bond] = self.get_prob(bond)
        self.probabilities = probs

    def set_one_bond(self) -> Bond:
        """
        Sets one bond of the lattice. First finds the maximum probability range. Then loops through
        bonds until the chosen probability is found

        Ex. Bond13 has a probability of 0.5, and Bonds[1-12] have a total probability summing to 3.6.
        Then Bond13 owns values from 3.6 to 4.1
        """
        assert len(self.probabilities) > 0
        max_probability = sum(self.probabilities.values())
        picked_probability = np.random.uniform(0, max_probability)

        # Find the bond that "owns" this probability
        chosen_bond: Optional[Bond] = None
        current_prob: float = 0.0
        items = list(self.probabilities.items())
        random.shuffle(items)
        for bond, prob in items:
            current_prob += prob
            if picked_probability < current_prob or np.isclose(picked_probability, current_prob):  # Match!
                chosen_bond = bond
                break
        assert chosen_bond is not None
        chosen_bond.add_bond()
        self.update(chosen_bond)
        return chosen_bond

    def update(self, chosen_bond: Bond):
        """
        Updates the probabilities after the [chosen_bond] has been added
        """
        # First remove from the dictionary (it has been set)
        del self.probabilities[chosen_bond]
        # Now updated all relevant bonds
        updated_bonds_list = self.updated_bonds(chosen_bond)
        for bond in updated_bonds_list:
            self.probabilities[bond] = self.get_prob(bond)
