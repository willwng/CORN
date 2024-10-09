"""
visualize_lattice.py allows visualizations of the position of the lattice
"""

import matplotlib.pyplot as plt
import numpy as np


class VisualizerParameters:
    draw_nodes: bool
    draw_bonds: bool
    draw_edge: bool

    bond_color: str
    node_color: str

    hor_shear: float

    def __init__(self, draw_nodes: bool, draw_bonds: bool, draw_pbc: bool, bond_color: str, node_color: str,
                 hor_shear: float):
        self.draw_nodes = draw_nodes
        self.draw_bonds = draw_bonds
        self.draw_edge = draw_pbc
        self.bond_color = bond_color
        self.node_color = node_color
        self.hor_shear = hor_shear


def plot_bonds(pos_matrix, i, j, lattice, color):
    thickness: float = 1 if lattice.get_length() < 200 else 0.3
    plt.plot([pos_matrix[i, 0], pos_matrix[j, 0]], [pos_matrix[i, 1], pos_matrix[j, 1]],
             color=color, linewidth=thickness)


class Visualizer:
    params: VisualizerParameters

    def __init__(self, params: VisualizerParameters):
        self.params = params

    def create_plot_bounds(self, lattice):
        x_min = -2 + -1.2 * lattice.get_length() * self.params.hor_shear
        x_max = (lattice.get_length() + lattice.get_height() * self.params.hor_shear) * 1.2
        y_min = -0.1 * lattice.get_height()
        y_max = 1.1 * lattice.get_height()
        plot_bounds = (x_min, x_max, y_min, y_max)
        return plot_bounds

    def draw_nodes(self, pos_matrix):
        plt.scatter(pos_matrix[:, 0], pos_matrix[:, 1], s=0.5, c=self.params.node_color)

    def visualize(self, lattice, pos_matrix, filename, rigid_edges=None, largest_cluster=None):
        """
        Creates a plot of the lattice, saved at filename

        :param pos_matrix: position matrix of each node
        :type pos_matrix: numpy vector of length (num_nodes*2)
        :param lattice: lattice structure
        :type lattice: (inherits AbstractLattice)
        :param filename: location to save plot
        :type filename: string
        :param rigid_edges: (optional) array containing the cluster labels for each bond
        :type rigid_edges: array of ints
        :param largest_cluster: array containing all bonds in the largest connecting cluster
        :type largest_cluster: List of Bond instances
        :return (void) Saves figure at filename
        """
        # Configure matplotlib
        if rigid_edges is None:
            rigid_edges = []
        plt.clf()
        plt.axis("scaled")
        plot_bounds = self.create_plot_bounds(lattice)
        plt.xlim(plot_bounds[0], plot_bounds[1])
        plt.ylim(plot_bounds[2], plot_bounds[3])

        pos_matrix = pos_matrix.reshape((-1, 2))
        active_bonds = lattice.get_active_bonds()
        active_bond_indices = np.zeros((len(active_bonds), 4), dtype=np.int32)
        for i, bond in enumerate(active_bonds):
            active_bond_indices[i][0] = bond.get_node1().get_id()
            active_bond_indices[i][1] = bond.get_node2().get_id()
            active_bond_indices[i][2] = bond.is_hor_pbc()
            active_bond_indices[i][3] = bond.is_top_pbc()

        i, j = active_bond_indices[:, 0], active_bond_indices[:, 1]
        idx_in = np.logical_and(active_bond_indices[:, 2] == 0, active_bond_indices[:, 3] == 0)
        idx_hor = active_bond_indices[:, 2] == 1
        idx_top = active_bond_indices[:, 3] == 1

        if self.params.draw_bonds:
            # Draw inner bonds
            i_in, j_in = i[idx_in], j[idx_in]
            plot_bonds(pos_matrix, i_in, j_in, lattice, color=self.params.bond_color)

        if self.params.draw_nodes:
            self.draw_nodes(pos_matrix)

        try:
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filename)
            print("Saved visualization at", filename)
        except PermissionError:
            print("Could not save visualization:", filename)
