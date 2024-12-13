"""
A double lattice consists of two of the same lattice types, one on top of the other
    (with one having different mechanical properties)
"""
import math
from typing import Optional

from lattice.abstract_lattice import AbstractLattice
from lattice.bond import Bond
from lattice.triangular_lattice import TriangularLattice


class DoubleTriangularLattice(AbstractLattice):
    def __init__(self, length: int, height: float):
        # Required to satisfy the AbstractLattice interface
        self.length = length
        self.height = height
        self.nodes = []
        self.bonds = []
        self.pi_bonds = []
        self.name = "DoubleTriangular"
        self.height_increment = math.sqrt(3.0) / 2.0

        # Initialize two triangular lattices
        self.network1 = TriangularLattice(length, height)
        self.network2 = TriangularLattice(length, height)
        self.set_height(self.network1.height)

        # Keep track of which bonds/pi bonds belong to which lattice
        self.bonds1, self.bonds2 = [], []
        self.pi_bonds1, self.pi_bonds2 = [], []

        # Combine the two networks
        self.coalesce_networks()

    def coalesce_networks(self) -> None:
        # Map from node index to the nodes in the first network
        idx_to_node = {node.get_id(): node for node in self.network1.get_nodes()}

        # Fix all the bonds in the second network to use the nodes from the first network
        for bond in self.network2.get_bonds():
            node1, node2 = bond.get_nodes()
            node1_id, node2_id = node1.get_id(), node2.get_id()
            mapped_node1, mapped_node2 = idx_to_node[node1_id], idx_to_node[node2_id]
            assert mapped_node1.get_xy() == node1.get_xy()
            assert mapped_node2.get_xy() == node2.get_xy()
            bond.n_1, bond.n_2 = mapped_node1, mapped_node2

        for pi_bond in self.network2.get_pi_bonds():
            vertex, e1, e2 = pi_bond.vertex, pi_bond.edge1, pi_bond.edge2
            vertex_id, e1_id, e2_id = vertex.get_id(), e1.get_id(), e2.get_id()
            mapped_vertex, mapped_e1, mapped_e2 = idx_to_node[vertex_id], idx_to_node[e1_id], idx_to_node[e2_id]
            pi_bond.vertex, pi_bond.edge1, pi_bond.edge2 = mapped_vertex, mapped_e1, mapped_e2

        # We are free to drop all nodes in the second network
        self.network2.nodes = []

        # Only keep the nodes from the first network
        self.nodes = self.network1.get_nodes()
        self.bonds = self.network1.get_bonds() + self.network2.get_bonds()
        self.pi_bonds = self.network1.get_pi_bonds() + self.network2.get_pi_bonds()

    def drop_bond(self, bond: Bond) -> None:
        if bond in self.network1.get_bonds():
            self.network1.drop_bond(bond)
        elif bond in self.network2.get_bonds():
            self.network2.drop_bond(bond)
        else:
            raise ValueError("Bond not found in either network")
        self.bonds.remove(bond)

    def get_removable_bonds(self) -> list[Bond]:
        return self.network2.get_bonds()

    def get_bond_occupations(self):
        """
        Returns p1, p2, the bond occupation fractions for the two networks
        """
        active1, bonds1 = self.network1.get_bond_occupation()
        active2, bonds2 = self.network2.get_bond_occupation()
        return active1 / bonds1, active2 / bonds2

    def update_active_bonds(self) -> None:
        self.network1.update_active_bonds()
        self.network2.update_active_bonds()
        self.active_bonds = self.network1.get_active_bonds().union(self.network2.get_active_bonds())
        self.update_active_pi_bonds()

    def update_active_pi_bonds(self) -> None:
        self.network1.update_active_pi_bonds()
        self.network2.update_active_pi_bonds()
        self.active_pi_bonds = self.network1.get_active_pi_bonds() + self.network2.get_active_pi_bonds()

    def set_stretch_mod(self, stretch_mod: float, stretch_mod2: Optional[float] = None) -> None:
        assert stretch_mod2 is not None
        self.network1.set_stretch_mod(stretch_mod)
        self.network2.set_stretch_mod(stretch_mod2)

    def set_bend_mod(self, bend_mod: float, bend_mod2: Optional[float] = None) -> None:
        assert bend_mod2 is not None
        self.network1.set_bend_mod(bend_mod)
        self.network2.set_bend_mod(bend_mod2)

    def set_tran_mod(self, tran_mod: float, tran_mod2: Optional[float] = None) -> None:
        assert tran_mod2 is not None
        self.network1.set_tran_mod(tran_mod)
        self.network2.set_tran_mod(tran_mod2)
