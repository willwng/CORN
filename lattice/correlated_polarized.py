"""
This class extends the Abstract lattice class

CorrelatedPolarizedLattice is a class that allows for 
correlated and polarized network generation
"""

from math import sqrt
from random import random, choice
from typing import List, Dict

from lattice.abstract_lattice import AbstractLattice
from lattice.bond import Bond
from lattice.node import Node
from lattice.pi_bond import PiBond
from lattice.bond_set_helper import BondSetter


class CorrelatedPolarizedLattice(AbstractLattice):
    # Default max number of neighbors for bonds, depends on lattice
    #    (for correlated networks)
    max_neighbors: int = 3
    # Maximum number of nearest neighbors (both directions)
    max_nearest_neighbors: int = 1
    # Spacing between each row of nodes, depends on lattice
    #   (for correlated networks)
    height_increment: float = 1
    generation_type: int = 0  # Network generation type, see parameters

    def set_bonds(
            self, prob_fill: float, strength: float = 0, rand: bool = False
    ) -> None:
        """
        Overrides the set_bonds method from Abstract Lattice
        :param prob_fill: target bond occupation probability
        :type prob_fill: float in [0, 1]
        :param strength: temporary variable holder
            (used for when correlated networks override this method)
        :type strength: float
        :type rand:
        """
        if self.generation_type == 0:
            super().set_bonds(prob_fill)
        elif self.generation_type == 1:
            self.set_bonds_pc_add(prob_fill, strength, 0)
        elif self.generation_type == 2:
            self.set_bonds_pc_remove(prob_fill, strength, 0)
        elif self.generation_type == 3:
            self.set_bonds_pc_add(prob_fill, strength, 1)
        elif self.generation_type == 4:
            self.set_bonds_pc_remove(prob_fill, strength, 1)
        elif self.generation_type == 5:
            self.set_bonds_super_remove(prob_fill)
        elif self.generation_type == 6:
            self.set_bonds_pc_add(prob_fill, strength, 2)
        elif self.generation_type == 7:
            self.set_bonds_pc_remove(prob_fill, strength, 2)
        elif self.generation_type == 8:
            self.set_bonds_distribution(prob_fill, frac_right=strength)
        elif self.generation_type == 9:
            self.set_bond_strands()
        elif self.generation_type == 10:
            self.set_bonds_par_nnn(prob_fill, strength)
        elif self.generation_type == 11:
            self.set_bonds_par_nnn_voting(prob_fill, strength, rule=4)
        elif self.generation_type == 12:
            self.set_bonds_par_nnn_voting(prob_fill, strength, rule=2)
        elif self.generation_type == 13:
            self.set_bonds_global_polarization(prob_fill, strength)
        else:
            print(f"Unknown generation type: {self.generation_type}")
            exit(1)

    def remove_bond(self, num_remove: int, strength: float = 0, *args, **kwargs) -> None:
        if self.generation_type != 13:
            super().remove_bond(num_remove)
        else:
            removable_bonds = [bond for bond in self.get_bonds() if not bond.is_boundary() and bond.exists()]
            if len(removable_bonds) < num_remove:
                for bond in removable_bonds:
                    bond.remove_bond()
                    self.active_bonds.remove(bond)
                return
            bonds_removed = 0
            while bonds_removed < num_remove:
                bond = choice(removable_bonds)
                if bond.is_boundary():
                    assert False
                if not bond.exists():
                    continue
                prob_fill = self.get_bond_global_polarization_prob(bond, strength)
                if random() > prob_fill:
                    bond.remove_bond()
                    self.active_bonds.remove(bond)
                    bonds_removed += 1
        return

    def add_bond(self, num_add: int, strength: float = 0, *args, **kwargs) -> None:
        if self.generation_type != 13:
            super().add_bond(num_add)
        else:
            addable_bonds = [bond for bond in self.get_bonds() if not bond.is_boundary() and not bond.exists()]
            if len(addable_bonds) < num_add:
                for bond in addable_bonds:
                    bond.add_bond()
                    self.active_bonds.append(bond)
                return
            bonds_added = 0
            while bonds_added < num_add:
                bond = choice(addable_bonds)
                if bond.is_boundary():
                    assert False
                if bond.exists():
                    continue
                prob_fill = self.get_bond_global_polarization_prob(bond, strength)
                if random() < prob_fill:
                    bond.add_bond()
                    self.active_bonds.append(bond)
                    bonds_added += 1
        return

    def create_top_pbc(self) -> None:
        """
        Creates the top periodic boundary conditions by removing the top layer
        of nodes and generating temporary bonds
        """
        # Remove top layer of nodes (if they exist)
        max_height = max(node.get_xy()[1] for node in self.nodes)
        self.nodes = [node for node in self.nodes if not (node.is_boundary() and node.get_xy()[1] == max_height)]
        # re-update the bonds (slow but ensures no extraneous/leftover bonds)
        self.generate_bonds()
        self.add_temp_bonds()
        # re-generate pi-bonds
        self.pi_bonds = []
        self.generate_pi_bonds()
        self.add_corner_pbc()

    def remove_top_pbc(self) -> None:
        """
        Removes the top periodic boundary conditions and replaces it with
            'real nodes and bonds'
        """
        # Remove all corners
        self.bonds = [bond for bond in self.bonds if not bond.is_corner()]
        self.pi_bonds = [pi_bond for pi_bond in self.pi_bonds if
                         not (pi_bond.get_bond1().is_corner() or pi_bond.get_bond2().is_corner())]
        # Add the top nodes and get the corresponding 'real' node
        # Get the height of the current top row (this is our second row) and calculate our new top row
        max_height = max(node.get_xy()[1] for node in self.nodes)
        new_row_height = max_height + self.height_increment
        # What number to start counting/identifying our new nodes
        node_id = max(node.get_id() for node in self.nodes) + 1
        # This maps the real/model nodes (on the bottom layer) to the newly generated nodes
        node_dictionary = {}
        last_old_node_idx = len(self.nodes) - 1
        # -- Start generating the top layer of nodes -- #
        last_new_node = None
        for i in range(1, self.length + 1, 1):
            second_node = Node(i, new_row_height, node_id)
            # The series of horizontal bonds on the top (boundary bonds)
            if last_new_node is not None:
                new_bond = self.create_bond(second_node, last_new_node)
                new_bond.set_boundary()
                second_node.set_boundary()
                last_new_node.set_boundary()

            last_new_node = second_node
            node_id += 1
            self.nodes.append(second_node)
            # Additionally, we will try to map the model node (which is on the bottom level)
            #   to the new node we just created (they have the same x-value)
            for n in self.nodes:
                if n.get_xy() == (i, 0):
                    model_node = n
                    assert model_node is not None
                    node_dictionary[model_node] = second_node
                    break

        # One periodic boundary condition - First new node to last new node
        assert last_new_node is not None
        first_new_node = self.nodes[last_old_node_idx + 1]
        new_bond = self.create_bond(last_new_node, first_new_node)
        new_bond.set_boundary()
        new_bond.set_hor_pbc()
        last_new_node.set_edge()
        last_new_node.set_boundary()
        first_new_node.set_edge()
        first_new_node.set_boundary()

        # This periodic boundary condition is weird and depends on the type of lattice
        # For Kagome, it is the bond connecting the first node of the second row (from top)
        # to the last node of the top row
        # for node in self.nodes:
        #     if node.get_xy()[1] == max_height:
        # new_bond = self.create_bond(last_node, node)
        # new_bond.set_boundary()
        # new_bond.set_edge()
        # last_node.set_edge()
        # last_node.set_boundary()
        # first_node.set_edge()
        # first_node.set_boundary()
        # break

        # For triangular, it is the first node of the top row to the last of the second row
        second_layer_nodes = []
        for second_node in self.nodes:
            if second_node.get_xy()[1] == max_height:
                second_layer_nodes.append(second_node)
        last_second_node = max(second_layer_nodes, key=lambda node: node.get_xy()[0])
        new_bond = self.create_bond(last_second_node, first_new_node)
        new_bond.set_hor_pbc()
        last_second_node.set_edge()
        first_new_node.set_edge()

        # Make a replacement bond for the ones connected to fake nodes
        for bond in self.bonds:
            if bond.is_temporary():
                # Get the lower node (this is the "real/model" node to make our new one)
                node1 = bond.get_node1()
                node2 = bond.get_node2()
                if node1.get_xy()[1] == 0:
                    model_node = node1
                    second_node = node2
                elif node2.get_xy()[1] == 0:
                    model_node = node2
                    second_node = node1
                else:
                    print("Impossible - no model node found")
                    assert False
                new_node = node_dictionary[model_node]
                self.create_bond(new_node, second_node, bond.exists())
        self.remove_temp_bonds()
        self.generate_pi_bonds()
        return

    def add_corner_pbc(self) -> None:
        # For triangular lattices, we need the bottom-left and top-right to be connected
        last_node = max(self.nodes, key=lambda node: node.get_id())
        first_node = min(self.nodes, key=lambda node: node.get_id())
        corner_bond = self.create_bond(first_node, last_node)
        corner_bond.set_temporary()

        def find_relevant_bond(node):
            bonds = [bond for bond in self.bonds if bond.get_node1() == node or bond.get_node2() == node]
            relevant_bonds = [bond for bond in bonds if
                              bond.get_direction() == 1 and not bond.is_hor_pbc() and not bond.is_temporary()]
            assert len(relevant_bonds) == 1
            return relevant_bonds[0]

        # Now figure out the two pi-bonds we need to add
        first_relevant_bond = find_relevant_bond(first_node)
        last_relevant_bond = find_relevant_bond(last_node)
        pi_bond_1 = PiBond(first_relevant_bond, corner_bond, v=first_node,
                           e1=first_relevant_bond.get_other_node(first_node), e2=last_node)
        pi_bond_2 = PiBond(last_relevant_bond, corner_bond, v=last_node,
                           e1=last_relevant_bond.get_other_node(last_node), e2=first_node)
        self.pi_bonds.append(pi_bond_1)
        self.pi_bonds.append(pi_bond_2)
        return

    def add_temp_bonds(self) -> None:
        num_nodes = len(self.nodes)
        max_height = max(node.get_xy()[1] for node in self.nodes)
        # Loop through possible combinations of 2 nodes
        for i in range(num_nodes):
            for j in range(0, i):
                node_1 = self.nodes[i]
                node_2 = self.nodes[j]
                # Get the coordinates of the two nodes and distance
                cor_1 = node_1.get_xy()
                cor_2 = node_2.get_xy()
                # Temporary Bonds: (Periodic boundary condition from top to
                # bottom) for network generation
                if (cor_1[1] == max_height and cor_2[1] == 0) or (cor_1[1] == 0 and cor_2[1] == max_height):
                    # Distance should compare in x-direction (y is simply height increment)
                    edge_distance = (cor_1[0] - cor_2[0]) ** 2 + self.height_increment ** 2
                    if edge_distance < 1.01:
                        bond = self.create_bond(node_1, node_2)
                        bond.set_temporary()
        return

    def remove_temp_bonds(self) -> None:
        # Ensure these bonds don't interfere
        for bond in self.bonds:
            if bond.is_temporary():
                bond.remove_bond()
        # Remove instances of where these temporary bonds could exist
        for node in self.node_bonds_dic.keys():
            bonds = self.node_bonds_dic[node]
            self.node_bonds_dic[node] = [bond for bond in bonds if not bond.is_temporary()]
        # Update bonds and pi bonds array
        self.bonds = [bond for bond in self.bonds if not bond.is_temporary()]
        self.pi_bonds = [pi_bond for pi_bond in self.pi_bonds if
                         not (pi_bond.get_bond1().is_temporary() or pi_bond.get_bond2().is_temporary())]

        return

    def update_neighbors(self) -> None:
        """
        update_neighbors is used for correlated network generation. It updates
        the 'neighbor' field in nodes
        """
        for node in self.nodes:
            node.reset_neighbors()
        for bond in self.bonds:
            if bond.exists():
                node1 = bond.get_node1()
                node2 = bond.get_node2()
                node1.add_neighbor()
                node2.add_neighbor()

    def set_bonds_pc_add(self, prob_fill, strength, gen_type) -> None:
        """
        Sets the bonds to correspond to network of correlated or polarized

        :param prob_fill: target bond occupation
        :type prob_fill: float
        :param strength: strength of correlation or polarized
        :type strength: float (0 <= strength < 1)
        :param gen_type: network generation type
            (currently 0 = correlated, 1 = polarized,
            2 = polarized with next-nearest-neighbors)
        :type gen_type: int, 0, 1 or 2
        """
        self.create_top_pbc()
        # For higher bond probabilities, no need to run correlation algorithm
        if prob_fill == 1.0:
            super().set_bonds(prob_fill=prob_fill)
            self.remove_top_pbc()
            self.update_active_bonds()
            return
        # Generate a network with 1% filled in bonds
        super().set_bonds(prob_fill=0.01)
        self.update_neighbors()

        # Add necessary number of bonds
        addable_bonds = [b for b in self.bonds if not b.exists()]
        active_bonds, total_bonds = self.get_bond_occupation()
        desired_bonds = prob_fill * total_bonds
        # Create a dictionary for faster lookup for bond -> pi-bonds
        bond_pi_bond_dic = self.create_bond_pi_bond_dic()
        while active_bonds < desired_bonds:
            bond = choice(addable_bonds)
            # Bond already exists
            if bond.exists():
                assert False
            # Get the probability of the bond - depends on generation type
            prob_fill_bond = 0
            if gen_type == 0:  # Correlated
                prob_fill_bond = self.get_probability_correlated(bond, strength)
            elif gen_type == 1:  # Polarized
                prob_fill_bond = self.get_probability_polarized(
                    bond, strength, bond_pi_bond_dic
                )
            elif gen_type == 2:
                prob_fill_bond = self.get_probability_nnn(bond, strength, bond_pi_bond_dic)
            # Roll a die to see if bond gets filled
            if random() < prob_fill_bond:
                bond.add_bond()
                active_bonds += 1
                addable_bonds.remove(bond)
                # Update neighbors for correlated networks
                if gen_type == 0:
                    bond.get_node1().add_neighbor()
                    bond.get_node2().add_neighbor()
        # Remember to keep the class invariant
        self.remove_top_pbc()
        self.update_active_bonds()
        return

    def set_bonds_pc_remove(self, prob_fill, strength, gen_type) -> None:
        """
        The same as set_bonds_pc_add, except that bonds are
        removed instead of added
        """
        # Create the periodic boundary condition from top to bottom and
        # set the bottom as boundary
        self.create_top_pbc()

        if prob_fill == 1.0:
            super().set_bonds(prob_fill=prob_fill)
            self.remove_top_pbc()
            self.update_active_bonds()
            return
        # Generate a network with 99% filled in bonds
        super().set_bonds(prob_fill=0.99)
        self.update_neighbors()

        # Determine the number of active bonds and the necessary number of bonds to be removed
        removable_bonds = [b for b in self.bonds if b.exists() and not b.is_boundary()]
        active_bonds, total_bonds = self.get_bond_occupation()
        desired_bonds = prob_fill * total_bonds
        # Create a dictionary for faster lookup for bond -> pi-bonds
        bond_pi_bond_dic = self.create_bond_pi_bond_dic()
        while active_bonds > desired_bonds:
            bond = choice(removable_bonds)  # Pick a random bond
            # Bond is already removed, no need to go further.
            # Also, boundary bonds should not be removed
            if not bond.exists() or bond.is_boundary():
                assert False
            # Get the probability of the bond - depends on generation type
            prob_fill_bond = 0
            if gen_type == 0:  # Correlated
                prob_fill_bond = self.get_probability_correlated(bond, strength)
            elif gen_type == 1:  # Polarized
                prob_fill_bond = self.get_probability_polarized(
                    bond, strength, bond_pi_bond_dic
                )
            elif gen_type == 2:  # Polarized w nnn
                prob_fill_bond = self.get_probability_nnn(bond, strength, bond_pi_bond_dic)
            # Roll a die to see if bond gets removed
            if strength == 0:
                prob_fill_bond = 0
            if random() < (1 - prob_fill_bond):
                bond.remove_bond()
                active_bonds -= 1
                removable_bonds.remove(bond)
                # Update neighbors for correlated networks
                if gen_type == 0:
                    bond.get_node1().remove_neighbor()
                    bond.get_node2().remove_neighbor()
        # Remember to keep the class invariant
        self.remove_top_pbc()
        self.update_active_bonds()
        return

    def get_probability_correlated(self, target_bond, corr_strength) -> float:
        """
        get_probability_correlated finds the probability of filling a bond
        based on the neighboring bonds to the two incident sites

        :param target_bond: desired bond to fill
        :param corr_strength: strength of correlation (0 <= corr_strength < 1)
        :return: probability of filling the bond
        """
        node1 = target_bond.get_node1()
        node2 = target_bond.get_node2()
        neighbors = node1.get_num_neighbors() + node2.get_num_neighbors()
        # Do not double-count the incident nodes. The bond already exists so
        # there are 2 extra neighbors
        if target_bond.exists():
            neighbors = neighbors - 2
        prob_fill = (1 - corr_strength) ** (self.max_neighbors - neighbors)
        return prob_fill

    def get_probability_polarized(
            self,
            target_bond: Bond,
            strength: float,
            bond_pi_bond_dic: Dict[Bond, List[PiBond]],
    ) -> float:
        """
        get_neighbors_polarized finds the probability of filling a bond based
        on the neighbors that are co-linear with the target bond by taking
        advantage of the pi-bonds

        :param target_bond: desired bond to fill
        :param strength: strength of polarization (0 <= pol_strength < 1)
        :return: probability of filling the bond
        """
        num_neighbors = self.get_polarized_neighbors(target_bond, bond_pi_bond_dic)
        # 2* max neighbors (because looking both directions)
        prob_fill = (1 - strength) ** (2 - num_neighbors)
        assert 0 <= num_neighbors <= 2
        return prob_fill

    def get_polarized_neighbors(self, target_bond, bond_pi_bond_dic: Dict[Bond, List[PiBond]]) -> int:
        neighbors = 0
        corr_pi_bonds = bond_pi_bond_dic[target_bond]
        for pi_bond in corr_pi_bonds:
            bond1 = pi_bond.get_bond1()
            bond2 = pi_bond.get_bond2()
            neighbor = bond2 if target_bond == bond1 else bond1
            if neighbor.exists():
                neighbors += 1
        return neighbors

    def get_polarized_neighbors_list(self, target_bond, bond_pi_bond_dic: Dict[Bond, List[PiBond]]) -> List[Bond]:
        neighbors = []
        corr_pi_bonds = bond_pi_bond_dic[target_bond]
        for pi_bond in corr_pi_bonds:
            bond1 = pi_bond.get_bond1()
            bond2 = pi_bond.get_bond2()
            neighbor = bond2 if target_bond == bond1 else bond1
            if not neighbor.exists():
                neighbors.append(neighbor)
        return neighbors

    def set_bonds_super_remove(self, prob_fill) -> None:
        """
        Remove bonds that have no co-linear neighbor (if no bonds exist,
                randomly pick one)
        """
        # Create the periodic boundary condition from top to bottom and set
        # the bottom as boundary
        self.create_top_pbc()

        if prob_fill == 1.0:
            super().set_bonds(prob_fill=prob_fill)
            self.remove_top_pbc()
            self.update_active_bonds()
            return
        # Generate a network with 99% filled in bonds
        super().set_bonds(prob_fill=0.99)
        # Determine the number of active bonds and the necessary number of
        # bonds to be removed
        active_bonds, total_bonds = self.get_bond_occupation()
        desired_bonds = prob_fill * total_bonds

        removable_bonds = self.get_removable_bonds()
        while active_bonds > desired_bonds:
            # No removable bonds left (stuck), pick some fraction of bonds and
            # remove them
            if not removable_bonds:
                bonds_to_remove = 1 + int((active_bonds - desired_bonds) / 100)
                active_bonds -= bonds_to_remove
                self.remove_random_bonds(bonds_to_remove)
                removable_bonds = self.get_removable_bonds()

            bond = choice(removable_bonds)  # Pick a random bond
            # Bond is already removed, no need to go further.
            # Also, boundary bonds should not be removed
            if not bond.exists() or bond.is_boundary():
                assert False

            bond.remove_bond()
            active_bonds -= 1
            removable_bonds = self.get_removable_bonds()
        # Remember to keep the class invariant
        self.remove_top_pbc()
        self.update_active_bonds()
        return

    def remove_random_bonds(self, num: int) -> None:
        """
        From the active bonds, randomly remove (set inactive) one
        """
        active_bonds = []
        for bond in self.bonds:
            if bond.exists() and not bond.is_boundary():
                active_bonds.append(bond)
        for _ in range(num):
            random_bond = choice(active_bonds)
            random_bond.remove_bond()
            active_bonds.remove(random_bond)
        return

    def get_removable_bonds(self) -> List[Bond]:
        """
        Returns a list of  onds that can be removed (not boundary and exist)
        """
        removable_bonds = []
        for pi_bond in self.pi_bonds:
            bond1, bond2 = pi_bond.get_bond1(), pi_bond.get_bond2()
            # Boundary bonds aren't removable
            if bond1.is_boundary() or bond2.is_boundary():
                continue
            if bond1.exists() and not bond2.exists():
                removable_bonds.append(bond1)
            elif not bond1.exists() and bond2.exists():
                removable_bonds.append(bond2)
        return list(set(removable_bonds))

    def add_dic(self, key, value, dic) -> None:
        """
        Assuming dictionary with values as lists. Adds to value list if
        non-existent, otherwise, creates a new list
        """
        if key in dic:
            dic[key].append(value)
        else:
            dic[key] = [value]

    def create_bond_pi_bond_dic(self) -> Dict[Bond, List[PiBond]]:
        """
        Creates a dictionary that maps bonds (key) to the corresponding
        pi-bonds that contain that bond
        """
        bond_pi_bond_dic = {}  # map of each bond to the corresponding pi-bond
        for pi_bond in self.pi_bonds:
            bond1 = pi_bond.get_bond1()
            bond2 = pi_bond.get_bond2()
            self.add_dic(bond1, pi_bond, bond_pi_bond_dic)
            self.add_dic(bond2, pi_bond, bond_pi_bond_dic)

        return bond_pi_bond_dic

    def get_next_nearest_neighbors(
            self, bond: Bond, bond_pi_bond_dic: Dict[Bond, List[PiBond]]
    ) -> int:
        """
        Returns the number of next-nearest-neighbor bonds to this one
        """
        visited = {}
        return self.nnn_helper(bond, 0, bond_pi_bond_dic, visited)

    def nnn_helper(
            self,
            bond: Bond,
            depth: int,
            bond_pi_bond_dic: Dict[Bond, List[PiBond]],
            visited: Dict[Bond, bool],
    ) -> int:
        """
        A helper function that performs a DFS search to return the number
        of next-nearest neighbors of the bond. Stops search when a
        specific depth is reached or when there are no existing bonds
        on either side
        """
        # Reached the max number of neighbors (both directions)
        if depth == self.max_nearest_neighbors:
            return 0
        # Ensure this bond has been set to visited
        visited[bond] = True
        # Recurse through the neighbors
        neighbors = 0
        corr_pi_bonds = bond_pi_bond_dic[bond]
        # assert len(corr_pi_bonds) == 2
        for pi_bond in corr_pi_bonds:
            bond1 = pi_bond.get_bond1()
            bond2 = pi_bond.get_bond2()
            # Each bond has already been visited. Discard
            if bond1 in visited and bond2 in visited:
                continue
            # Find this corresponding bond and the neighboring bond
            neighbor = bond2 if bond == bond1 else bond1
            if neighbor.exists():
                # Add 1 for the neighbor, plus all of the neighbor's neighbors
                neighbors += 1 + self.nnn_helper(
                    neighbor, depth + 1, bond_pi_bond_dic, visited
                )
            # Keep traversing even if strand ends (can turn off)
            else:
                neighbors += self.nnn_helper(
                    neighbor, depth + 1, bond_pi_bond_dic, visited
                )
        return neighbors

    def get_probability_nnn(
            self,
            target_bond: Bond,
            strength: float,
            bond_pi_bond_dic: Dict[Bond, List[PiBond]],
    ) -> float:
        """
        get_probability_correlated finds the probability of filling a bond
        based on the next nearest neighbor bonds

        :param target_bond: desired bond to fill
        :return: probability of filling the bond
        """
        nearest_neighbors = self.get_next_nearest_neighbors(
            target_bond, bond_pi_bond_dic
        )
        # 2* max neighbors (because looking both directions)
        prob_fill = (1 - strength) ** (
                2 * self.max_nearest_neighbors - nearest_neighbors
        )
        prob_fill = (
                sqrt(nearest_neighbors / (2 * self.max_nearest_neighbors)) ** strength
        )
        # print(prob_fill)
        return prob_fill

    def set_bond_strands(self) -> None:
        """
        Creates a network with complete strands of bonds
        """
        boundary_bonds = 0
        addable_bonds: List[Bond] = []
        for b in self.bonds:
            if b.is_boundary():
                b.add_bond()
                boundary_bonds += 1
            else:
                b.remove_bond()
                addable_bonds.append(b)

        bond_dic = self.create_bond_pi_bond_dic()
        starting_bonds = [2020]  # 180 for 10x11
        seen = {}
        for start in starting_bonds:
            self.dfs_add(addable_bonds[start], bond_dic, seen)
        self.update_active_bonds()

    def dfs_add(
            self, bond: Bond, bond_dic: Dict[Bond, List[PiBond]], seen: Dict[Bond, bool]
    ):
        seen[bond] = True
        corr_pi_bonds = bond_dic[bond]
        # assert len(corr_pi_bonds) == 2
        bond.add_bond()
        for pi_bond in corr_pi_bonds:
            bond1 = pi_bond.get_bond1()
            bond2 = pi_bond.get_bond2()
            # Each bond has already been visited. Discard
            if bond1 in seen and bond2 in seen:
                continue
            # Find this corresponding bond and the neighboring bond
            neighbor = bond2 if bond == bond1 else bond1
            self.dfs_add(neighbor, bond_dic, seen)
        return

    def create_bond_hor_dic(self) -> Dict[Bond, List[Bond]]:
        """
        Creates a dictionary that maps bonds (key) to the corresponding
        neighboring bonds with the same horizontal and direction
        """
        bond_hor_bond_dic = {}
        # Create lookup by (direction, height)
        dir_hor_dic = {}
        for bond in self.bonds:
            key = (bond.get_higher_pos(), bond.get_direction())
            self.add_dic(key, bond, dir_hor_dic)
        # Find neighbors
        dirs = [-1, 1]
        hor_space = 2  # For Kagome
        ver_space = sqrt(3)
        for bond in self.bonds:
            for dir in dirs:
                pos = bond.get_higher_pos()
                if bond.get_direction() == 0:  # horizontal
                    pos_n = (pos[0], pos[1] + dir * ver_space)
                else:
                    pos_n = (pos[0] + dir * hor_space, pos[1])
                key_n = (pos_n, bond.get_direction())
                if key_n in dir_hor_dic:
                    self.add_dic(bond, dir_hor_dic[key_n][0], bond_hor_bond_dic)
        return bond_hor_bond_dic

    def get_probability_par_nnn(
            self,
            target_bond: Bond,
            strength: float,
            bond_pi_bond_dic: Dict[Bond, List[PiBond]],
            bond_hor_dic: Dict[Bond, List[Bond]],
            rule: int
    ) -> float:
        """
        get_probability_correlated finds the probability of filling a bond
        based on the nearest neighbors and the horizontal neighbors

        :param target_bond: desired bond to fill
        :return: probability of filling the bond
        """
        num_neighbors = self.get_polarized_neighbors(target_bond, bond_pi_bond_dic)
        # num_neighbors = self.get_next_nearest_neighbors(target_bond, bond_pi_bond_dic)
        if target_bond not in bond_hor_dic or rule == 2:
            hor_neighbors = 0
            max_hor_neighbors = 0
        else:
            active_hor = [a for a in bond_hor_dic[target_bond] if a.exists()]
            hor_neighbors = len(active_hor)
            max_hor_neighbors = 2
        # 2* max neighbors (because looking both directions)
        prob_fill = (1 - strength) ** (
                (2 * self.max_nearest_neighbors + max_hor_neighbors) - num_neighbors - hor_neighbors)
        # print(prob_fill)
        return prob_fill

    def set_bonds_par_nnn(self, prob_fill, strength) -> None:
        """
        Sets the bonds to correspond to correlated network which considers
        the horizontal nearest neighbors

        :param prob_fill: target bond occupation
        :type prob_fill: float
        :param strength: strength of correlation or polarized
        :type strength: float (0 <= strength < 1)
        """
        self.create_top_pbc()
        # For higher bond probabilities, no need to run correlation algorithm
        if prob_fill == 1.0:
            super().set_bonds(prob_fill=prob_fill)
            self.remove_top_pbc()
            self.update_active_bonds()
            return
        # Generate a network with 1% filled in bonds
        super().set_bonds(prob_fill=0.01)
        # Add necessary number of bonds
        addable_bonds = [b for b in self.bonds if not b.exists()]
        active_bonds, total_bonds = self.get_bond_occupation()
        desired_bonds = prob_fill * total_bonds
        # Create a dictionary for faster lookup for bond -> pi-bonds
        bond_pi_bond_dic = self.create_bond_pi_bond_dic()
        bond_hor_dic = self.create_bond_hor_dic()
        while active_bonds < desired_bonds:
            bond = choice(addable_bonds)
            # Bond already exists
            if bond.exists():
                assert False
            # Get the probability of the bond - depends on generation type
            prob_fill_bond = self.get_probability_par_nnn(bond, strength, bond_pi_bond_dic, bond_hor_dic, rule=2)
            # Roll a die to see if bond gets filled
            if random() < prob_fill_bond:
                bond.add_bond()
                active_bonds += 1
                addable_bonds.remove(bond)
        # Remember to keep the class invariant
        self.remove_top_pbc()
        self.update_active_bonds()
        return

    def get_par_nnn_updated(
            self,
            target_bond: Bond,
            bond_pi_bond_dic: Dict[Bond, List[PiBond]],
            bond_hor_dic: Dict[Bond, List[Bond]],
            rule: int,
    ) -> List[Bond]:
        to_update_list = self.get_polarized_neighbors_list(
            target_bond, bond_pi_bond_dic
        )
        if target_bond in bond_hor_dic:
            if rule == 4:
                active_hor = [a for a in bond_hor_dic[target_bond] if not a.exists()]
                to_update_list.extend(active_hor)
        return to_update_list

    def set_bonds_par_nnn_voting(self, prob_fill: float, strength: float, rule: int = 2) -> None:
        """
        Sets the bonds to correspond to correlated network which considers
        the horizontal nearest neighbors

        :param prob_fill: target bond occupation
        :type prob_fill: float
        :param strength: strength of correlation or polarized
        :type strength: float (0 <= strength < 1)
        :param rule: Which "generation rule to use"
        :type rule: int, typically 2 or 4
        """
        self.create_top_pbc()
        # For higher bond probabilities, no need to run correlation algorithm
        if prob_fill == 1.0:
            super().set_bonds(prob_fill=prob_fill)
            self.remove_top_pbc()
            self.update_active_bonds()
            return
        super().set_bonds(prob_fill=0.01)
        # Add necessary number of bonds
        active_bonds, total_bonds = self.get_bond_occupation()
        desired_bonds = prob_fill * total_bonds
        # Helper dictionaries
        bond_pi_bond_dic = self.create_bond_pi_bond_dic()
        bond_hor_dic = self.create_bond_hor_dic()

        # Helper functions
        def get_prob(b: Bond) -> float:
            return self.get_probability_par_nnn(b, strength, bond_pi_bond_dic, bond_hor_dic, rule)

        def get_updated(b: Bond) -> List[Bond]:
            return self.get_par_nnn_updated(b, bond_pi_bond_dic, bond_hor_dic, rule)

        bond_setter = BondSetter(self.bonds, get_prob, get_updated)
        while active_bonds < desired_bonds:
            bond_setter.set_one_bond()
            active_bonds += 1
        # Remember to keep the class invariant
        self.remove_top_pbc()
        self.update_active_bonds()
        return

    def get_bond_global_polarization_prob(self, bond: Bond, strength: float) -> float:
        """
        Returns the probability of a bond being filled in the global polarization
        model

        :param strength: strength of correlation or polarized
        :type strength: float (0 <= strength < 1)
        :param bond: bond to get probability of
        :return: probability of a bond being filled
        :rtype: float
        """
        # Get the probability of the bond
        direction = bond.get_direction()
        if direction == 0:
            prob_fill = 1 - strength
        elif direction == 1:  # right-leaning bonds
            prob_fill = strength
        else:
            prob_fill = 1 - strength
        return prob_fill

    def set_bonds_global_polarization(self, prob_fill, strength) -> None:
        """
        Sets the bonds to correspond to network of correlated or polarized

        :param prob_fill: target bond occupation
        :type prob_fill: float
        :param strength: strength of correlation or polarized
        :type strength: float (0 <= strength < 1)
        """
        # For higher bond probabilities, no need to run correlation algorithm
        if prob_fill == 1.0:
            super().set_bonds(prob_fill=prob_fill)
            self.update_active_bonds()
            return
        # Generate a network with 0% filled in bonds
        super().set_bonds(prob_fill=0.0)
        addable_bonds = [bond for bond in self.bonds if not bond.exists() and not bond.is_boundary()]
        # Add necessary number of bonds
        active_bonds, total_bonds = self.get_bond_occupation()
        desired_bonds = prob_fill * total_bonds
        while active_bonds < desired_bonds:
            bond = choice(addable_bonds)
            # Bond already exists
            if bond.exists():
                assert False
            # Get the probability of the bond
            prob_fill_bond = self.get_bond_global_polarization_prob(bond, strength)

            if random() < prob_fill_bond:
                bond.add_bond()
                active_bonds += 1
                addable_bonds.remove(bond)
        # Remember to keep the class invariant
        self.update_active_bonds()
        return
