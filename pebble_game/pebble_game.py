"""
pebble_game.py is used to 'play the pebble game' based on
'An Algorithm for Two Dimensional Rigidity Percolation: The Pebble Game'
Donald J. Jacobs and Bruce Hendrickson

The code also takes inspiration from https://github.com/CharlesLiu7/Pebble-Game
for find_pebbles and rearrange_pebbles methods
"""
import copy
from enum import Enum


class Label(Enum):
    """
    class Label is used to create enum and allow for more readable and easier to edit code
    """
    RIGID = 'RIGID'
    FLOPPY = 'FLOPPY'
    PINNED = 'PINNED'


class PebbleGame:
    def __init__(self, lattice):
        n = len(lattice.get_nodes())
        self.n = n
        # Each node has two pebbles
        self.pebbles = [[None, None] for _ in range(n)]
        self.seen = [False for _ in range(n)]
        self.path = [-1 for _ in range(n)]
        self.edges = []
        self.rigid_nodes = [False for _ in range(self.n)]
        self.laman_nodes = [False for _ in range(self.n)]  # Whether the node belongs to a laman subgraph
        self.independent_edges = []  # Contains basis of independent edges
        self.redundant_edges = []  # Redundant edges that are in overconstrained sub_graphs

        bonds = lattice.get_bonds()
        for bond in bonds:
            if bond.exists():
                n_1 = bond.get_node1().get_id()
                n_2 = bond.get_node2().get_id()
                self.edges.append((n_1, n_2))

    def simple_game(self, lattice):
        self.__init__(lattice)
        print("Running pebble game")
        for a, b in self.edges:
            self.enlarge_cover(a, b)

        leftover_pebbles = 0
        for p in self.pebbles:
            leftover_pebbles += p.count(None)
        print(leftover_pebbles, 'leftover pebbles')

        return copy.deepcopy(self.pebbles)

    def create_edge_basis(self, lattice):
        self.__init__(lattice)
        for a, b in self.edges:
            # Copy the old pebbles - requires deep copy
            old_pebbles = copy.deepcopy(self.pebbles)
            # G_4e, try adding edge 4. If 4 pebbles are obtained, edge is independent
            added = True
            for pebbles_found in range(4):
                added = self.enlarge_cover(a, b)
                if not added:
                    # Check Lemma 2.6: If the new edge, e, is tripled instead of quadrupled to form G3e, then G3e has
                    #   a pebble covering
                    assert pebbles_found == 3
                    # A bond placed between a pair of sites is redundant when two free pebbles cannot be obtained
                    #    at each incident site
                    self.redundant_edges.append((a, b))
                    # The set of sites visited in the failed search comprise a Laman subgraph
                    self.laman_nodes = [(r or n) for r, n in zip(self.rigid_nodes, self.seen)]

            # Restore the old pebbles, then add the edge to the basis if the edge is independent (Theorem 2.8)
            self.pebbles = copy.deepcopy(old_pebbles)
            if added:
                self.enlarge_cover(a, b)
                self.independent_edges.append((a, b))
        return

    def get_floppy_modes(self, lattice):
        self.create_edge_basis(lattice)
        floppy_modes = 2 * self.n - (len(self.edges) - len(self.redundant_edges))
        return floppy_modes

    def find_rigid_edges(self, lattice):
        print("Finding rigid clusters")
        # Place the independent bonds in the network
        self.create_edge_basis(lattice)
        # All bonds and sites are initially unlabeled
        edge_labels = [None for _ in range(len(self.edges))]
        site_labels = [None for _ in range(self.n)]
        cluster_label = 0

        # Look through the unlabeled bonds
        for edge_num, (edge, e_label) in enumerate(zip(self.edges, edge_labels)):
            if e_label is None:
                # Step 1: Introduce a new cluster label for an unlabeled bond.
                cluster_label += 1
                edge_labels[edge_num] = cluster_label
                a, b = edge[0], edge[1]  # Two incident sites
                # Remember the old pebbles (pinning pebbles is only temporary)
                old_pebbles = copy.deepcopy(self.pebbles)
                # Sites that are encountered in a pebble rearrangement
                visit_rearrange = [False for _ in range(self.n)]
                # Step 2. Gather three pebbles at its two incident sites and temporarily pin the three free pebbles down
                for _ in range(3):
                    added = self.enlarge_cover(a, b)
                    assert added
                    self.fix_duplicates(old_pebbles)  # Remove the instances of two pebbles on the same edge
                    for p in self.path:
                        if p != -1:
                            visit_rearrange[p] = True

                # Step 3. Mark the two incident sites as rigid.
                site_labels[a] = Label.RIGID
                site_labels[b] = Label.RIGID
                rigid_nodes = [a, b]

                # Step 4: Using the network, neighbor table, check the unmarked nearest neighbors to the set of
                #   rigid sites
                for node in rigid_nodes:
                    assert site_labels[node] == Label.RIGID
                    for neighbor in self.pebbles[node]:
                        self.step_four(neighbor, site_labels, visit_rearrange, rigid_nodes)
                # All neighbors to the rigid_nodes should be labeled (floppy or rigid)
                assert self.check_neighbors(rigid_nodes, site_labels)

                # Step 8: All bonds between pairs of sites marked rigid are given the same cluster label.
                for i, e in enumerate(self.edges):
                    if site_labels[e[0]] == Label.RIGID and site_labels[e[1]] == Label.RIGID:
                        edge_labels[i] = cluster_label
                # Step 9: Remove floppy and rigid marks from all sites (nodes)
                site_labels = [None for _ in range(self.n)]
                # The pinned pebbles can now be freed
                self.pebbles = copy.deepcopy(old_pebbles)
        bond_labels = self.label_all_bonds(lattice, edge_labels)
        return bond_labels

    def label_all_bonds(self, lattice, edge_labels):
        bonds = lattice.get_bonds()
        all_labels = [-1 for _ in range(len(bonds))]
        count = 0
        for i, bond in enumerate(bonds):
            if bond.exists():
                all_labels[i] = edge_labels[count]
                count += 1
        return all_labels

    def fix_duplicates(self, old_pebbles):
        for v in range(len(self.pebbles)):
            if self.pebbles[v][0] == self.pebbles[v][1]:
                if self.pebbles[v][0] != old_pebbles[v][0]:
                    self.pebbles[v][0] = old_pebbles[v][0]
                else:
                    self.pebbles[v][0] = old_pebbles[v][1]

    def check_neighbors(self, rigid_nodes, site_labels):
        # It is ok for neighbors to be rigid since they are already part of the set of rigid nodes
        # It should not be the case that the neighbor is unlabeled
        for node in rigid_nodes:
            for neighbor in self.pebbles[node]:
                if neighbor is not None and neighbor != Label.PINNED and site_labels[neighbor] is None:
                    return False
        return True

    def step_four(self, neighbor, site_labels, visit_rearrange, rigid_nodes):
        # Neighbor exists (not a free or pinned pebble) and is currently unlabeled rigid or floppy
        if neighbor is not None and neighbor is not Label.PINNED and site_labels[neighbor] is None:
            # For each new unmarked site, perform a pebble search and attempt to free a pebble.
            self.seen = [False for _ in range(len(self.seen))]
            self.path = [-1 for _ in range(len(self.path))]
            found_pebble = self.find_pebble(neighbor, None)
            # If a free pebble is found; the site is not mutually rigid with respect to the initial bond
            #   nor is any other site that was encountered during a pebble rearrangement. Mark all
            #   these sites floppy
            if found_pebble:
                site_labels[neighbor] = Label.FLOPPY
                for i, s in enumerate(visit_rearrange):
                    if s:
                        site_labels[i] = Label.FLOPPY
            # If a free pebble is not found; the site is mutually rigid with respect to the initial bond
            #   as well as all other sites that make up the failed search. Mark these sites as rigid and
            #   include them in the set of rigid sites
            else:
                for i, s in enumerate(self.seen):
                    if s:
                        rigid_nodes.append(i)
                        site_labels[i] = Label.RIGID
        return

    def enlarge_cover(self, a, b):
        self.seen = [False for _ in range(len(self.seen))]
        self.path = [-1 for _ in range(len(self.path))]

        found = self.find_pebble(a, b)

        if found:
            self.rearrange_pebbles(a, b)
            return True

        if not self.seen[b]:
            found = self.find_pebble(b, a)
            if found:
                self.rearrange_pebbles(b, a)
                return True
        return False

    def find_pebble(self, v, _v):
        self.seen[v] = True
        self.path[v] = -1
        # If v has a free pebble
        if None in self.pebbles[v]:
            return True
        else:
            # x = neighbor along edge my pebble covers
            x = self.pebbles[v][0]
            if x != Label.PINNED and not self.seen[x] and x != _v:
                self.path[v] = x
                found = self.find_pebble(x, _v)
                if found:
                    return True
            # y = neighbor along edge my other pebble covers
            y = self.pebbles[v][1]
            if y != Label.PINNED and not self.seen[y] and y != _v:
                self.path[v] = y
                found = self.find_pebble(y, _v)
                if found:
                    return True
        return False

    def rearrange_pebbles(self, v, _v):
        # If path is -1, that means the pebbles of v have not been allocated. Allocate a pebble to the new edge
        # The following has been added to the algorithm outlined to allow for adding of edges
        if self.path[v] == -1:
            if self.pebbles[v][0] is None:
                self.pebbles[v][0] = _v
            elif self.pebbles[v][1] is None:
                self.pebbles[v][1] = _v
            else:
                print('No path, but no pebbles')
            return
        v_copy = v
        # Otherwise, v does not have free pebbles, so we look to its path
        while self.path[v] != -1:
            w = self.path[v]
            # w has a free pebble, use it
            # Cover edge (v,w) with pebble from w
            if self.path[w] == -1:
                if self.pebbles[w][0] is None:
                    self.pebbles[w][0] = v
                elif self.pebbles[w][1] is None:
                    self.pebbles[w][1] = v
            else:
                # Cover edge (v,w) with pebble from edge (w, path(w))
                _w = self.path[w]
                if self.pebbles[w][0] == _w:
                    self.pebbles[w][0] = v
                elif self.pebbles[w][1] == _w:
                    self.pebbles[w][1] = v
            v = w

        # Path is not -1 (no free pebbles), allocate for the new bond (v, _v)
        # Use the pebble that was originally used for (v, path[v])
        if self.pebbles[v_copy][0] == self.path[v_copy]:
            self.pebbles[v_copy][0] = _v
        if self.pebbles[v_copy][1] == self.path[v_copy]:
            self.pebbles[v_copy][1] = _v
        return
