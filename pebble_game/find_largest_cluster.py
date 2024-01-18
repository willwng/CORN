"""
find_largest_cluster.py is used to find the largest connected cluster in a network
The function implements a DFS to find connected components

"""


class LargestCluster:
    bond_neighbors = {}

    def add_to_neighbors(self, bond1, bond2):
        self.bond_neighbors[bond1].append(bond2)
        self.bond_neighbors[bond2].append(bond1)

    def __init__(self, lattice):
        max_id = max([node.get_id() for node in lattice.nodes])
        # array containing all the bonds attached to this node
        bonds_connected_node = [[] for _ in range(max_id + 1)]
        bonds = lattice.get_bonds()
        for bond in bonds:
            self.bond_neighbors[bond] = []
        # Find the neighboring bonds to each bond
        for i in range(len(bonds) - 1):
            bond1 = bonds[i]
            if not bond1.exists() or bond1.is_boundary():
                continue
            for j in range(i, len(bonds)):
                bond2 = bonds[j]
                if not bond2.exists() or bond2.is_boundary():
                    continue
                n_1, n_2 = bond1.get_node1(), bond1.get_node2()
                n_3, n_4 = bond2.get_node1(), bond2.get_node2()
                if n_1 == n_3 or n_1 == n_4 or n_2 == n_3 or n_2 == n_4:
                    self.add_to_neighbors(bond1, bond2)

        # for bond in bonds:
        #     # non-existent bonds and boundaries don't count
        #     if not bond.exists() or bond.is_boundary():
        #         continue
        #     node_1 = bond.get_node1().get_id()
        #     node_2 = bond.get_node2().get_id()
        #     bonds_connected_node[node_1].append(bond)
        #     bonds_connected_node[node_2].append(bond)
        #
        # bond_neighbors = {}
        # for node_bonds in bonds_connected_node:
        #     for i in range(0, len(node_bonds) - 1):
        #         for j in range(i, len(node_bonds)):
        #             bond1 = node_bonds[i]
        #             bond2 = node_bonds[j]
        #             if bond1 not in bond_neighbors:
        #                 bond_neighbors[bond1] = []
        #             else:
        #                 bond_neighbors[bond1].append(bond2)
        #             if bond2 not in bond_neighbors:
        #                 bond_neighbors[bond2] = []
        #             else:
        #                 bond_neighbors[bond2].append(bond1)
        # self.bond_neighbors = bond_neighbors

    def dfs(self, bond, visited):
        connected_components = []
        # Stack for dfs, add initial bond
        stack = [bond]
        # Add the initial bond
        while stack:
            current = stack.pop()
            visited[current] = True
            connected_components.append(current)
            # Add to stack to dfs further
            for neighbor in self.bond_neighbors[current]:
                if neighbor not in visited:
                    stack.append(neighbor)
        return connected_components

    def get_largest_cluster(self):
        largest_cluster = []
        visited = {}
        for bond in self.bond_neighbors:
            # If there are even neighbors to this bond
            if self.bond_neighbors[bond]:
                connected = self.dfs(bond, visited)
                if len(connected) > len(largest_cluster):
                    largest_cluster = connected
        return largest_cluster
