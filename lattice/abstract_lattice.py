"""
Abstract class for each lattice - contains shared methods and behaviors
A lattice contains information about its nodes, bonds, pi_bonds
The positions of each node is not changed - the lattice represents
the initial lattice
"""
from random import shuffle
from typing import Dict, List, Tuple, Optional

import numpy as np
import scipy.spatial as spatial

from lattice.bond import Bond
from lattice.node import Node
from lattice.pi_bond import PiBond


class AbstractLattice:
    # Size of network
    length: int = 0
    height: float = 0.0
    name: str = ""  # Name of lattice (Kagome, Triangular, Square)

    # Amount of spacing between each row of nodes, depends on lattice (should override)
    height_increment: float = 1.0

    # ALl the node, bond, and pi-bond instances
    nodes: List[Node] = []
    bonds: List[Bond] = []
    pi_bonds: List[PiBond] = []

    # Contains all instances of only existing bonds and boundary bonds (faster lookup)
    active_bonds: set[Bond] = set()
    boundary_bonds: List[Bond] = []
    active_pi_bonds: List[PiBond] = []

    # Used as a lookup for node -> bonds to speed up pi-bond generation
    node_bonds_dic: Dict[Node, List[Bond]] = {}

    # The following are all getter methods, based according to their name
    def get_length(self) -> int:
        return self.length

    def get_height(self) -> float:
        return self.height

    def set_length(self, length) -> None:
        self.length = length

    def set_height(self, height) -> None:
        self.height = height

    def get_bonds(self) -> List[Bond]:
        return self.bonds

    def get_active_bonds(self) -> set[Bond]:
        return self.active_bonds

    def get_nodes(self) -> List[Node]:
        return self.nodes

    def get_pi_bonds(self) -> List[PiBond]:
        return self.pi_bonds

    def get_active_pi_bonds(self) -> List[PiBond]:
        return self.active_pi_bonds

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        """ Returns a more descriptive name than [get_name]"""
        return f"{self.get_name()}_{self.get_length()}_{int(round(self.get_height(), 0))}"

    def __hash__(self):
        return hash(id(self))

    def get_max_node_id(self):
        return max(node.get_id() for node in self.nodes)

    def update_active_bonds(self) -> None:
        """
        Updates the array of active bonds
        """
        self.active_bonds = set([bond for bond in self.bonds if bond.exists()])
        self.update_active_pi_bonds()

    def remove_bond(self, bond: Bond) -> None:
        """
        Deactivate a bond from the lattice (does not remove the object, just sets it inactive)
        """
        bond.set_inactive()
        self.active_bonds.remove(bond)
        self.update_active_pi_bonds()

    def update_active_pi_bonds(self) -> None:
        """
        Updates the array of active bonds
        """
        self.active_pi_bonds = [pi_bond for pi_bond in self.pi_bonds if pi_bond.exists()]

    def drop_bond(self, bond: Bond) -> None:
        """
        Used for *entirely removing* a bond from the lattice, not just setting inactive
        Only for temporary or extraneous bonds
        """
        bond.set_inactive()
        self.bonds.remove(bond)
        self.pi_bonds = [pi_bond for pi_bond in self.pi_bonds if
                         pi_bond.get_bond1() != bond and pi_bond.get_bond2() != bond]
        self.update_active_bonds()

    def add_node_bonds_dic(self, node: Node, bond: Bond) -> None:
        if node in self.node_bonds_dic:
            self.node_bonds_dic[node].append(bond)
        else:
            self.node_bonds_dic[node] = []  # Allocate new array
            self.node_bonds_dic[node].append(bond)
        return

    def create_bond(self, node_1: Node, node_2: Node, active: bool = True) -> Bond:
        """
        Factored out code to be used in generating bonds
        :param node_1: First node
        :param node_2: Second node
        :param active: Whether the bond is active or not
        :return: The created bond
        """
        bond = Bond(node_1, node_2, active)
        self.bonds.append(bond)
        # Add to lookup table
        self.add_node_bonds_dic(node_1, bond)
        self.add_node_bonds_dic(node_2, bond)
        return bond

    def generate_nodes(self) -> List[Node]:
        """
        Generates the nodes within the lattice
        """
        raise NotImplementedError("Subclasses must implement this method")

    def generate_bonds(self) -> None:
        """
        Generates the bonds between each pair of nodes that have a distance of 1.0
        (including periodic boundary conditions)
        This current implementation is O(d*n*log(n)), d = dimension, n = number of nodes

        Take note of the following invariant:
            - if a bond is a horizontal PBC, then the x position of node 1 is greater than node 2
            - if a bond is a top PBC, then the y position of node 1 is greater than node 2
        """
        self.bonds = []  # Reset the bonds
        self.node_bonds_dic = {}  # Reset lookup table
        num_nodes = len(self.nodes)

        # Build a simple list of the node positions and a lookup table
        node_positions = np.zeros((num_nodes, 2))
        node_lookup = []
        for node in sorted(self.nodes, key=lambda x: x.get_id()):
            node_positions[node.get_id()] = node.get_xy()
            node_lookup.append(node)
        node_positions = np.array(node_positions)

        # Build a KDTree for fast nearest neighbor lookup
        tree = spatial.cKDTree(node_positions)

        # Regular bonds
        for i in range(num_nodes):
            node_i = node_lookup[i]
            j_nodes = np.array(tree.query_ball_point(node_positions[i], 1.01, 2))
            # Look at bond locations where i > j
            j_nodes = j_nodes[np.where(i > j_nodes)]
            for j in j_nodes:
                node_j = node_lookup[j]
                self.create_bond(node_i, node_j)

        # Periodic boundary conditions. Try shifting every node to the left: only edge nodes will be within distance
        shifted_pos = node_positions - np.array([self.get_length(), 0])
        for i in range(num_nodes):
            node_i = node_lookup[i]
            j_nodes = np.array(tree.query_ball_point(shifted_pos[i], 1.01, 2))
            for j in j_nodes:
                node_j = node_lookup[j]
                bond = self.create_bond(node_i, node_j)
                bond.set_hor_pbc()

        # Periodic boundary conditions. Try shifting every node to the down: only edge nodes will be within distance
        max_height = max(node.get_xy()[1] for node in self.nodes)
        shifted_pos = node_positions - np.array([0, max_height + self.height_increment])
        for i in range(num_nodes):
            node_i = node_lookup[i]
            j_nodes = np.array(tree.query_ball_point(shifted_pos[i], 1.01, 2))
            for j in j_nodes:
                node_j = node_lookup[j]
                bond = self.create_bond(node_i, node_j)
                bond.set_top_pbc()

        # Finally, create a bond from the bottom left to top-right
        max_length = max(node.get_xy()[0] for node in self.nodes)
        node_i = [node for node in self.nodes if node.get_xy()[0] == max_length and node.get_xy()[1] == max_height][0]
        node_j = [node for node in self.nodes if node.get_xy()[0] == 0 and node.get_xy()[1] == 0][0]
        bond = self.create_bond(node_i, node_j)
        bond.set_hor_pbc(), bond.set_top_pbc()
        print("Generated " + str(len(self.bonds)) + " bonds")

    def set_bonds(self, prob_fill: float) -> None:
        """
        Adds/removes each bond with probability of prob_fill or (1-prob_fill).
        Should be called after the lattice object has been entirely generated (all bonds and pi bonds generated).

        :param prob_fill: probability of a bond existing
        :type prob_fill: float in range [0, 1]
        """
        # First set all bonds to inactive
        for bond in self.bonds:
            bond.set_inactive()

        # The number of desired bonds to be added (total num bonds - num boundary bonds)
        desired_bonds = int(prob_fill * len(self.bonds))

        # Randomly shuffle the list and pick the bonds to be removed
        addable_bonds = [bond for bond in self.get_bonds()]
        shuffle(addable_bonds)
        for i in range(desired_bonds):
            addable_bonds[i].set_active()

        # Maintain class invariant
        self.update_active_bonds()

    def set_stretch_mod(self, stretch_mod: float, stretch_mod2: Optional[float] = None) -> None:
        """
        Sets the stretching modulus for all bonds to [stretch_mod]
        """
        for bond in self.bonds:
            bond.stretch_mod = stretch_mod

    def set_bend_mod(self, bend_mod: float, bend_mod2: Optional[float] = None) -> None:
        """
        Sets the bending modulus for all pi-bonds to [bend_mod]
        """
        for pi_bond in self.pi_bonds:
            pi_bond.set_bend_mod(bend_mod)

    def set_tran_mod(self, tran_mod: float, tran_mod2: Optional[float] = None) -> None:
        """
        Sets the transverse modulus for all bonds to [tran_mod]
        """
        for bond in self.bonds:
            bond.tran_mod = tran_mod

    def get_bond_occupation(self) -> Tuple[int, int]:
        """
        Returns a tuple, containing the number of active bonds
        (not including top/bottom boundary bonds) and the number of total bonds
        (not including boundary)

        :return: A tuple (number active, total number)
            excluding the boundary bonds
        """
        return len(self.active_bonds), len(self.bonds)

    def get_base_lattice(self) -> Tuple[List[Node], List[Bond], List[PiBond]]:
        """
        Returns the necessary data to create the same lattice (not for pickling).
        Used for lattice re-use where runs use the same size lattice
        """
        return self.nodes, self.bonds, self.pi_bonds

    def get_lattice_data(self):
        """
        Returns the necessary data to create the same lattice after pickling
        and reloading
        """
        # Position of all the nodes
        node_pos_data = []
        # Boundary, edge, has_bonds of each node
        node_data = []
        for node in self.nodes:
            node_pos_data.append(node.get_xy())
            node_data.append((node.is_boundary(), node.is_edge(), node.is_bonded()))
        # The ids of the incident nodes that each bond points to
        bond_node_data = []
        for bond in self.bonds:
            n1_id = bond.get_node1().get_id()
            n2_id = bond.get_node2().get_id()
            bond_node_data.append((n1_id, n2_id))
        # Boundary, Edge, Activation of each bond
        bond_data = []
        bond_index = {}
        for i, bond in enumerate(self.bonds):
            bond_data.append((bond.is_hor_pbc(), bond.exists()))
            bond_index[bond] = i

        pi_bond_data = []
        for pi_bond in self.pi_bonds:
            b1_id = bond_index[pi_bond.get_bond1()]
            b2_id = bond_index[pi_bond.get_bond2()]
            vertex = pi_bond.get_vertex_node().get_id()
            edge1, edge2 = map(lambda n: n.get_id(), pi_bond.get_edge_nodes())
            pi_bond_data.append((b1_id, b2_id, vertex, edge1, edge2))
        return node_pos_data, node_data, bond_node_data, bond_data, pi_bond_data

    def set_all_bonds_active(self) -> None:
        """
        Sets all bonds to active
        """
        for bond in self.bonds:
            bond.set_active()
        self.update_active_bonds()
        return

    def patch_pbc(self) -> None:
        """
        Patches the periodic boundary conditions for the lattice
        Slight hacky-fix. Remove bonds with double "imaginary positions"
            (i.e. bonds that are both periodic in x and y)
        """
        imag_x = set([bond.get_node1() for bond in self.bonds if bond.is_hor_pbc()])
        imag_y = set([bond.get_node1() for bond in self.bonds if bond.is_top_pbc()])
        for bond in [bond for bond in self.bonds if bond.is_hor_pbc() or bond.is_top_pbc()]:
            node_1, node_2 = bond.get_node1(), bond.get_node2()
            if node_1 in (imag_y.difference(imag_x)) and node_2 in imag_x:
                self.drop_bond(bond)
            if node_1 in (imag_y.intersection(imag_x)) and node_2 not in (
                    imag_y.union(imag_x)) and node_2.get_id() != 0:
                self.drop_bond(bond)
        return

    def load_lattice(self, node_pos_data, node_data, bond_node_data, bond_data, pi_bond_data) -> None:
        """
        Loads a lattice from bond_dat (typically obtained through pickling get_bond_data())
        :param node_pos_data: array containing (x, y) of nodes in order of id's
        :type node_pos_data: 1-D array of tuples
        :param node_data: array containing tuples of 3 booleans whether
            corresponding bond is boundary, edge, has bonds
        :type node_data: 1-D array of tuples containing 3 booleans
        :param bond_node_data: array containing the ids of the incident nodes
            that each bond points to
        :type bond_node_data: 1-D array of tuples
        :param bond_data: array containing tuples of 3 booleans whether
            corresponding bond is boundary, edge, exists
        :param pi_bond_data: array containing tuples of 2 integers corresponding to the indices of the bonds,
            and 3 integers corresponding to the indices of the nodes
        :type bond_data: 1-D array of tuples containing 3 booleans
        """
        # Load the node positions and assign unique ids
        self.nodes = []
        self.length, self.height = 0, 0
        for i, (x, y) in enumerate(node_pos_data):
            self.nodes.append(Node(x, y, i))
            self.length = int(max(self.length, x))
            self.height = max(self.length, y)
        # Load the bonds by their incident node ids
        self.bonds = []
        for (n1_id, n2_id) in bond_node_data:
            n1 = self.nodes[n1_id]
            n2 = self.nodes[n2_id]
            self.bonds.append(Bond(n1, n2, True))
        # Set the bond data (boundary, edge, activation)
        for b, (boundary, edge, exists) in zip(self.bonds, bond_data):
            if edge:
                b.set_hor_pbc()
            if not exists:
                b.set_inactive()
        # Load the pi bonds by their incident bond ids and node ids
        for b1_id, b2_id, vertex, edge1, edge2 in pi_bond_data:
            b1 = self.bonds[b1_id]
            b2 = self.bonds[b2_id]
            self.pi_bonds.append(PiBond(b1, b2, self.nodes[vertex], self.nodes[edge1], self.nodes[edge2]))
        # Generate the needed pi bonds and then maintain class invariant
        self.update_active_bonds()
        return

    def check_bonds_colinear(self, bond_i: Bond, bond_j: Bond, vertex: Node):
        """
        Checks if two bonds are co-linear
        """
        # Find the unit vector of each bond
        pos_i1, pos_i2 = bond_i.get_xy_xy()
        r_i = np.subtract(pos_i2, pos_i1)
        if bond_i.is_hor_pbc():
            r_i[0] = r_i[0] + self.length
        if bond_i.is_top_pbc():
            r_i[1] = r_i[1] + self.height + self.height_increment
        r_i = r_i / np.linalg.norm(r_i, axis=0)

        # Vector from vertex node to edge (if sign_i is 1). Flipped if sign_i is -1
        sign_i = 1 if bond_i.get_node1() == vertex else -1
        r_i = sign_i * r_i

        pos_j1, pos_j2 = bond_j.get_xy_xy()
        r_j = np.subtract(pos_j2, pos_j1)
        if bond_j.is_hor_pbc():
            r_j[0] = r_j[0] + self.length
        if bond_j.is_top_pbc():
            r_j[1] = r_j[1] + self.height + self.height_increment
        r_j = r_j / np.linalg.norm(r_j, axis=0)

        sign_j = 1 if bond_j.get_node1() == vertex else -1
        r_j = sign_j * r_j

        angle = np.arctan2(np.cross(r_i, r_j), np.dot(r_i, r_j))
        if angle < 0:
            angle = angle + 2 * np.pi
        return abs(abs(angle) - np.pi) < 0.01

    def generate_pi_bonds(self) -> None:
        """
        Generates the pi bonds in the lattice (series of two co-linear bonds with a node vertex)
        """
        self.pi_bonds = []
        # If the pre-made dictionary doesn't exist, use a brute force method
        if not self.node_bonds_dic:
            self.brute_generate_pi_bonds()
            return

        # Otherwise we can use the dictionary that maps nodes to the bonds that contain them
        #   (essentially iterating over possible vertices)
        for node in self.node_bonds_dic.keys():
            bonds = self.node_bonds_dic[node]
            # Iterate through possible pairs of bonds that contain [node]
            for i in range(len(bonds) - 1):
                bond_i = bonds[i]
                for j in range(i + 1, len(bonds)):
                    bond_j = bonds[j]

                    # Define vertex and edge nodes
                    vertex = node
                    e1 = (bond_i.get_node1() if node != bond_i.get_node1() else bond_i.get_node2())
                    e2 = (bond_j.get_node1() if node != bond_j.get_node1() else bond_j.get_node2())

                    if self.check_bonds_colinear(bond_i, bond_j, vertex):
                        pi_bond = PiBond(bond_i, bond_j, vertex, e1, e2)
                        self.pi_bonds.append(pi_bond)
        print("Generated " + str(len(self.pi_bonds)) + " pi bonds")

    def brute_generate_pi_bonds(self) -> None:
        """
        Generates the pi bonds in the lattice
        (series of two co-linear bonds with a node vertex)
        """
        self.pi_bonds = []
        # Loop through the bonds to find common vertices (node with two bonds)
        num_bonds = len(self.bonds)
        bonds = self.bonds
        # Iterate through pairs of bonds
        for i in range(num_bonds - 1):
            bond_i = bonds[i]
            for j in range(i + 1, num_bonds):
                bond_j = bonds[j]
                # Whether these two bonds are joined by a common node vertex
                found_common_vertex = False
                if bond_i.get_node1() == bond_j.get_node1():
                    found_common_vertex = True
                    vertex = bond_i.get_node1()
                    e1 = bond_i.get_node2()
                    e2 = bond_j.get_node2()
                elif bond_i.get_node2() == bond_j.get_node2():
                    found_common_vertex = True
                    vertex = bond_i.get_node2()
                    e1 = bond_i.get_node1()
                    e2 = bond_j.get_node1()
                elif bond_i.get_node1() == bond_j.get_node2():
                    found_common_vertex = True
                    vertex = bond_i.get_node1()
                    e1 = bond_i.get_node2()
                    e2 = bond_j.get_node1()
                elif bond_i.get_node2() == bond_j.get_node1():
                    found_common_vertex = True
                    vertex = bond_i.get_node2()
                    e1 = bond_i.get_node1()
                    e2 = bond_j.get_node2()
                # If no common vertex, continue
                if not found_common_vertex:
                    continue

                if self.check_bonds_colinear(bond_i, bond_j, vertex):
                    pi_bond = PiBond(bond_i, bond_j, vertex, e1, e2)
                    self.pi_bonds.append(pi_bond)
        print("Generated " + str(len(self.pi_bonds)) + " pi bonds")
