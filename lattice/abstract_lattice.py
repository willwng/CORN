"""
Abstract class for each lattice - contains shared methods and behaviors
A lattice contains information about its nodes, bonds, pi_bonds
The positions of each node is not changed - the lattice represents
the initial lattice
"""
import random
from typing import Dict, List, Tuple

from lattice.node import Node
from lattice.bond import Bond
from lattice.pi_bond import PiBond
from random import shuffle
import numpy as np
import scipy.spatial as spatial


class AbstractLattice:
    # Size of network
    length: int = 0
    height: float = 0.0
    name: str = ""  # Name of lattice (Kagome, Triangular, Square)

    # ALl the node, bond, and pi-bond instances
    nodes: List[Node] = []
    bonds: List[Bond] = []
    pi_bonds: List[PiBond] = []

    # Contains all instances of only existing bonds and boundary bonds (faster lookup)
    active_bonds: List[Bond] = []
    boundary_bonds: List[Bond] = []
    active_pi_bonds: List[PiBond] = []

    # Default max number of neighbors for bonds, depends on lattice
    # (used for correlated networks)
    max_neighbors: int = 6
    # Default spacing between each row of nodes, depends on lattice
    # (used for correlated networks)
    height_increment: int = 1
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

    def get_active_bonds(self) -> List[Bond]:
        return self.active_bonds

    def get_nodes(self) -> List[Node]:
        return self.nodes

    def get_pi_bonds(self) -> List[PiBond]:
        return self.pi_bonds

    def get_active_pi_bonds(self) -> List[PiBond]:
        return self.active_pi_bonds

    def drop_pi_bonds(self) -> None:
        self.pi_bonds = []

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
        self.active_bonds = [bond for bond in self.bonds if bond.exists()]
        self.update_active_pi_bonds()

    def update_boundary_bonds(self) -> None:
        """
        Updates the array of boundary bonds
        """
        self.boundary_bonds = [bond for bond in self.bonds if bond.is_boundary()]

    def update_active_pi_bonds(self) -> None:
        """
        Updates the array of active bonds
        """
        self.active_pi_bonds = [pi_bond for pi_bond in self.pi_bonds if pi_bond.exists()]

    def drop_bond(self, bond: Bond) -> None:
        bond.remove_bond()
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
        node_1.add_bond()
        node_2.add_bond()
        bond = Bond(node_1, node_2, active)
        self.bonds.append(bond)
        # Add to lookup table
        self.add_node_bonds_dic(node_1, bond)
        self.add_node_bonds_dic(node_2, bond)
        return bond

    def define_bounded_nodes(self, top: bool, bottom: bool, right: bool = False, left: bool = False) -> None:
        """
        Sets all the bonds at the top and bottom to be "bounded"
        such that their position is fixed during minimization
        """
        # Reset the boundary nodes
        [node.set_non_boundary() for node in self.nodes]
        min_x = min(node.get_xy()[0] for node in self.nodes)
        max_x = max(node.get_xy()[0] for node in self.nodes)
        min_y = min(node.get_xy()[1] for node in self.nodes)
        max_y = max(node.get_xy()[1] for node in self.nodes)
        for node in self.nodes:
            node_x, node_y = node.get_xy()
            if node_x == min_x and left:
                node.set_boundary()
            if node_x == max_x and right:
                node.set_boundary()
            if node_y == min_y and bottom:
                node.set_boundary()
            if node_y == max_y and top:
                node.set_boundary()
        self.update_boundary_bonds()

    def get_num_boundary_bonds(self) -> int:
        return sum(1 for b in self.bonds if b.is_boundary())

    def generate_bonds(self) -> None:
        """
        Generates the bonds between each pair of nodes that have a distance of 1.0
        (including periodic boundary conditions)
        Currently is O(d*n*log(n))

        We note of the following invariant:
        if a bond is a horizontal PBC, then the x position of node 1 is greater than node 2
        if a bond is a top PBC, then the y position of node 1 is greater than node 2
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
                node_i.set_edge()
                node_j.set_edge()
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
                node_i.set_top_edge()
                node_j.set_top_edge()
                bond = self.create_bond(node_i, node_j)
                bond.set_top_pbc()

        # Finally, create a bond from the bottom left to top-right
        max_length = max(node.get_xy()[0] for node in self.nodes)
        node_i = [node for node in self.nodes if node.get_xy()[0] == max_length and node.get_xy()[1] == max_height][0]
        node_j = [node for node in self.nodes if node.get_xy()[0] == 0 and node.get_xy()[1] == 0][0]
        node_i.set_edge(), node_i.set_top_edge(), node_j.set_edge(), node_j.set_top_edge()
        bond = self.create_bond(node_i, node_j)
        bond.set_hor_pbc(), bond.set_top_pbc()
        print("Generated " + str(len(self.bonds)) + " bonds")

    def remove_edges(self, horizontal: bool) -> List[Bond]:
        """
        Removes the PBC edges from the lattice (horizontal if horizontal, vertical if not)
        Returns the list of removed bonds
        """
        hor_edges = [bond for bond in self.get_bonds() if bond.is_hor_pbc() and bond.exists()]
        top_edges = [bond for bond in self.get_bonds() if bond.is_top_pbc() and bond.exists()]
        remove_edges = hor_edges if horizontal else top_edges
        for bond in remove_edges:
            bond.remove_bond()
        self.update_active_bonds()
        return remove_edges

    def remove_bounded_edges(self) -> List[Bond]:
        bounded_edges = [
            bond for bond in self.get_bonds() if
            bond.get_node1().is_boundary() and bond.get_node2().is_boundary()
            and bond.exists()
        ]
        for bond in bounded_edges:
            bond.remove_bond()
        self.update_active_bonds()
        return bounded_edges

    def restore_edges(self, remove_edges: List[Bond]) -> None:
        """
        Re-adds the edges that were removed
        """
        for bond in remove_edges:
            bond.add_bond()
        self.update_active_bonds()
        return

    def remove_bond(self, num_remove: int, *args, **kwargs) -> None:
        """
        Removes a random bond from this lattice
        """
        removable_bonds = [bond for bond in self.get_bonds() if not bond.is_boundary() and bond.exists()]
        # If there are not enough bonds to add, add all of them
        if len(removable_bonds) < num_remove:
            for bond in removable_bonds:
                bond.remove_bond()
                self.active_bonds.remove(bond)
        else:
            bonds_removed = 0
            while bonds_removed < num_remove:
                bond = random.choice(removable_bonds)
                bond.remove_bond()
                self.active_bonds.remove(bond)
                bonds_removed += 1
        # Maintain the active bonds
        return

    def add_bond(self, num_add: int, *args, **kwargs) -> None:
        """
        Removes a random bond from this lattice
        """
        addable_bonds = [bond for bond in self.get_bonds() if not bond.is_boundary() and not bond.exists()]
        # If there are not enough bonds to add, add all of them
        if len(addable_bonds) < num_add:
            for bond in addable_bonds:
                bond.add_bond()
                self.active_bonds.append(bond)
        else:
            bonds_added = 0
            while bonds_added < num_add:
                bond = random.choice(addable_bonds)
                bond.add_bond()
                self.active_bonds.append(bond)
                bonds_added += 1
        return

    def set_bonds(self, prob_fill: float, *args, **kwargs) -> None:
        """
        Adds/removes each bond with probability of prob_fill or (1-prob_fill).
        Should be called after the lattice object has been entirely generated (all bonds and pi bonds generated).

        :param prob_fill: probability of a bond existing
        :type prob_fill: float in range [0, 1]
        """
        # Boundary bonds cannot be removed, so we ensure they are added. All other bonds are removed temporarily
        boundary_bonds = 0
        addable_bonds = []
        for b in self.bonds:
            if b.is_boundary():
                b.add_bond()
                boundary_bonds += 1
            else:
                b.remove_bond()
                addable_bonds.append(b)

        # The number of desired bonds to be added (total num bonds - num boundary bonds)
        desired_bonds = int(prob_fill * (len(self.bonds) - boundary_bonds))
        active_bonds = 0

        # Randomly shuffle the list and pick the bonds to be removed
        shuffle(addable_bonds)
        current_bond = 0
        while active_bonds < desired_bonds:
            addable_bonds[current_bond].add_bond()
            current_bond += 1
            active_bonds += 1
        # Maintain class invariant
        self.update_active_bonds()

    def set_bonds_distribution(
            self, prob_fill: float, frac_right: float = 0.5
    ) -> None:
        """
        Sets the bonds in a lattice based on a distribution
        1/3 horizontal, frac_right right, (2/3 - frac_right) left

        :param prob_fill: probability of a bond existing
        :type prob_fill: float in range [0, 1]
        :param frac_right: fraction of right leaning bonds
        :type frac_right: float in range 0 <= 2/3
        """
        # Same as earlier
        boundary_bonds = 0
        addable_bonds: List[Bond] = []
        for b in self.bonds:
            if b.is_boundary():
                b.add_bond()
                boundary_bonds += 1
            else:
                b.remove_bond()
                addable_bonds.append(b)
        desired_bonds = int(prob_fill * (len(self.bonds) - boundary_bonds))
        desired_hor = int(desired_bonds / 3)
        desired_r = int(desired_bonds * frac_right)
        desired_l = desired_bonds - desired_hor - desired_r
        active_hor, active_r, active_l = 0, 0, 0

        # Randomly shuffle the list and pick the bonds to be removed
        shuffle(addable_bonds)
        current_bond = 0
        while active_hor < desired_hor or active_l < desired_l or active_r < desired_r:
            potential_bond = addable_bonds[current_bond]
            direction = potential_bond.get_direction()
            if direction == 0 and active_hor < desired_hor:
                potential_bond.add_bond()
                active_hor += 1
            elif direction == 1 and active_r < desired_r:
                potential_bond.add_bond()
                active_r += 1
            elif direction == 2 and active_l < desired_l:
                potential_bond.add_bond()
                active_l += 1
            current_bond += 1
            # current_bond += 1
        # Maintain class invariant
        self.update_active_bonds()

    def get_bond_occupation(self) -> Tuple[int, int]:
        """
        Returns a tuple, containing the number of active bonds
        (not including top/bottom boundary bonds) and the number of total bonds
        (not including boundary)

        :return: A tuple (number active, total number)
            excluding the boundary bonds
        """
        return len(self.active_bonds) - len(self.boundary_bonds), len(self.bonds) - len(self.boundary_bonds)

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
            bond_data.append((bond.is_boundary(), bond.is_hor_pbc(), bond.exists()))
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
            bond.add_bond()
        self.update_active_bonds()
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
        # Load the node data (boundary, edge, exists)
        for i, (boundary, edge, exists) in enumerate(node_data):
            node = self.nodes[i]
            if boundary:
                node.set_boundary()
            if edge:
                node.set_edge()
            if exists:
                node.add_bond()
        # Load the bonds by their incident node ids
        self.bonds = []
        for (n1_id, n2_id) in bond_node_data:
            n1 = self.nodes[n1_id]
            n2 = self.nodes[n2_id]
            self.bonds.append(Bond(n1, n2, True))
        # Set the bond data (boundary, edge, activation)
        for b, (boundary, edge, exists) in zip(self.bonds, bond_data):
            if boundary:
                b.set_boundary()
            if edge:
                b.set_hor_pbc()
            if not exists:
                b.remove_bond()
        # Load the pi bonds by their incident bond ids and node ids
        for b1_id, b2_id, vertex, edge1, edge2 in pi_bond_data:
            b1 = self.bonds[b1_id]
            b2 = self.bonds[b2_id]
            self.pi_bonds.append(PiBond(b1, b2, self.nodes[vertex], self.nodes[edge1], self.nodes[edge2]))
        # Generate the needed pi bonds and then maintain class invariant
        self.update_active_bonds()
        self.update_boundary_bonds()
        return

    def generate_pi_bonds(self) -> None:
        """
        Generates the pi bonds in the lattice
        (series of two colinear bonds with a node vertex)
        """
        self.pi_bonds = []
        # If dictionary doesn't exist, use a brute method
        if not self.node_bonds_dic:
            self.brute_generate_pi_bonds()
            return

        for node in self.node_bonds_dic.keys():
            bonds = self.node_bonds_dic[node]
            for i in range(len(bonds) - 1):
                bond_i = bonds[i]
                for j in range(i + 1, len(bonds)):
                    bond_j = bonds[j]
                    # Define vertex and edges
                    vertex = node
                    e1 = (bond_i.get_node1() if node != bond_i.get_node1() else bond_i.get_node2())
                    e2 = (bond_j.get_node1() if node != bond_j.get_node1() else bond_j.get_node2())

                    # Get positions of the two nodes per each of two bonds
                    pos1_i = bond_i.get_node1().get_xy()
                    pos2_i = bond_i.get_node2().get_xy()
                    pos1_j = bond_j.get_node1().get_xy()
                    pos2_j = bond_j.get_node2().get_xy()

                    # Find the unit vector of each bond
                    r_i = np.subtract(pos2_i, pos1_i)
                    r_j = np.subtract(pos2_j, pos1_j)

                    # Account for edge bonds being shorter in distance
                    if bond_i.is_hor_pbc():
                        if pos1_i[0] > pos2_i[0]:
                            r_i[0] = r_i[0] + self.length
                        else:
                            r_i[0] = r_i[0] - self.length
                    if bond_j.is_hor_pbc():
                        if pos1_j[0] > pos2_j[0]:
                            r_j[0] = r_j[0] + self.length
                        else:
                            r_j[0] = r_j[0] - self.length

                    # Account for top periodic bonds being shorter in distance
                    height_change = self.height + self.height_increment
                    if bond_i.is_temporary():
                        if pos1_i[1] > pos2_i[1]:
                            r_i[1] = r_i[1] + height_change
                        else:
                            r_i[1] = r_i[1] - height_change
                    if bond_j.is_temporary():
                        if pos1_j[1] > pos2_j[1]:
                            r_j[1] = r_j[1] + height_change
                        else:
                            r_j[1] = r_j[1] - height_change

                    # Normalize
                    r_i = r_i / np.linalg.norm(r_i, axis=0)
                    r_j = r_j / np.linalg.norm(r_j, axis=0)
                    # The two bonds should be co-linear
                    cos_phi = np.dot(r_i, r_j)
                    if abs(abs(cos_phi) - 1) < 0.01:
                        pi_bond = PiBond(bond_i, bond_j, vertex, e1, e2)
                        self.pi_bonds.append(pi_bond)
        print("Generated " + str(len(self.pi_bonds)) + " pi bonds")

    def brute_generate_pi_bonds(self) -> None:
        """
        Generates the pi bonds in the lattice
        (series of two colinear bonds with a node vertex)
        """
        self.pi_bonds = []
        # Loop through the bonds to find common vertices (node with two bonds)
        num_bonds = len(self.bonds)
        bonds = self.bonds
        vertices = 0
        vertex = None
        e1, e2 = None, None
        for i in range(num_bonds - 1):
            bond_i = bonds[i]
            for j in range(i + 1, num_bonds):
                bond_j = bonds[j]
                # Whether these two bonds are joined by a common node vertex
                is_vertex = False
                # First node of first bond is first node of second bond
                if bond_i.get_node1() == bond_j.get_node1():
                    is_vertex = True
                    vertex = bond_i.get_node1()
                    e1 = bond_i.get_node2()
                    e2 = bond_j.get_node2()
                # Second node of first bond is second node of second bond
                elif bond_i.get_node2() == bond_j.get_node2():
                    is_vertex = True
                    vertex = bond_i.get_node2()
                    e1 = bond_i.get_node1()
                    e2 = bond_j.get_node1()
                # First node of first bond is second node of second bond
                elif bond_i.get_node1() == bond_j.get_node2():
                    is_vertex = True
                    vertex = bond_i.get_node1()
                    e1 = bond_i.get_node2()
                    e2 = bond_j.get_node1()
                # Second node of first bond is first node of second bond
                elif bond_i.get_node2() == bond_j.get_node1():
                    is_vertex = True
                    vertex = bond_i.get_node2()
                    e1 = bond_i.get_node1()
                    e2 = bond_j.get_node2()
                if is_vertex:
                    vertices += 1
                    # Get positions of the two nodes per each of two bonds
                    pos1_i = bond_i.get_node1().get_xy()
                    pos2_i = bond_i.get_node2().get_xy()
                    pos1_j = bond_j.get_node1().get_xy()
                    pos2_j = bond_j.get_node2().get_xy()

                    # Find the unit vector of each bond
                    r_i = np.subtract(pos2_i, pos1_i)
                    r_j = np.subtract(pos2_j, pos1_j)

                    # Account for edge bonds being shorter in distance
                    if bond_i.is_hor_pbc():
                        if pos1_i[0] > pos2_i[0]:
                            r_i[0] = r_i[0] + self.length
                        else:
                            r_i[0] = r_i[0] - self.length
                    if bond_j.is_hor_pbc():
                        if pos1_j[0] > pos2_j[0]:
                            r_j[0] = r_j[0] + self.length
                        else:
                            r_j[0] = r_j[0] - self.length

                    # Account for top periodic bonds being shorter in distance
                    height_change = self.height + self.height_increment
                    if bond_i.is_temporary():
                        if pos1_i[1] > pos2_i[1]:
                            r_i[1] = r_i[1] + height_change
                        else:
                            r_i[1] = r_i[1] - height_change
                    if bond_j.is_temporary():
                        if pos1_j[1] > pos2_j[1]:
                            r_j[1] = r_j[1] + height_change
                        else:
                            r_j[1] = r_j[1] - height_change

                    # Normalize
                    r_i = r_i / np.linalg.norm(r_i, axis=0)
                    r_j = r_j / np.linalg.norm(r_j, axis=0)
                    # The two bonds should be co-linear
                    cos_phi = np.dot(r_i, r_j)
                    if abs(abs(cos_phi) - 1) < 0.01:
                        pi_bond = PiBond(bond_i, bond_j, vertex, e1, e2)
                        self.pi_bonds.append(pi_bond)
        print("Generated " + str(len(self.pi_bonds)) + " pi bonds")
        # print("Generated " + str(vertices) + " vertices")
