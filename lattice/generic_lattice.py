from lattice.abstract_lattice import AbstractLattice

import numpy as np


def make_generic(lattice: AbstractLattice, rng: np.random.Generator, d_shift: float) -> None:
    """
    Makes the lattice generic by shifting the nodes slightly
    """
    # Sort by id's to ensure replicable results
    sorted_nodes = sorted(lattice.get_nodes(), key=lambda n: n.get_id())
    for node in sorted_nodes:
        if node.is_boundary():
            continue
        x, y = node.get_xy()
        x += rng.uniform(-d_shift, d_shift)
        y += rng.uniform(-d_shift, d_shift)
        # x += d_shift * np.cos(rng.uniform(0, 2 * np.pi))
        # y += d_shift * np.sin(rng.uniform(0, 2 * np.pi))
        node.set_xy(x=x, y=y)

    return
