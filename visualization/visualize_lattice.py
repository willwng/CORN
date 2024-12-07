"""
visualize_lattice.py allows visualizations of the position of the lattice
"""
import os

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


def plot_bonds(pos_matrix, i, j, thickness, color, line_style):
    plt.plot([pos_matrix[i, 0], pos_matrix[j, 0]], [pos_matrix[i, 1], pos_matrix[j, 1]],
             color=color, linewidth=thickness, linestyle=line_style)


def draw_active_bonds(lattice, pos_matrix, thickness, color, line_style):
    # Gather only active bonds
    active_bonds = lattice.get_active_bonds()
    active_bond_indices = np.zeros((len(active_bonds), 4), dtype=np.int32)
    for i, bond in enumerate(active_bonds):
        active_bond_indices[i][0] = bond.get_node1().get_id()
        active_bond_indices[i][1] = bond.get_node2().get_id()
        active_bond_indices[i][2] = bond.is_hor_pbc()
        active_bond_indices[i][3] = bond.is_top_pbc()

    # Draw inner bonds only
    i, j = active_bond_indices[:, 0], active_bond_indices[:, 1]
    idx_in = np.logical_and(active_bond_indices[:, 2] == 0, active_bond_indices[:, 3] == 0)

    i_in, j_in = i[idx_in], j[idx_in]
    plot_bonds(pos_matrix, i_in, j_in, thickness=thickness, color=color, line_style=line_style)
    return


def create_plot_bounds(pos_matrix):
    x_min, x_max = np.min(pos_matrix[:, 0]), np.max(pos_matrix[:, 0])
    y_min, y_max = np.min(pos_matrix[:, 1]), np.max(pos_matrix[:, 1])
    # add some padding
    x_min -= 0.1 * (x_max - x_min)
    x_max += 0.1 * (x_max - x_min)
    y_min -= 0.1 * (y_max - y_min)
    y_max += 0.1 * (y_max - y_min)
    plot_bounds = (x_min, x_max, y_min, y_max)
    return plot_bounds


class Visualizer:
    params: VisualizerParameters

    def __init__(self, params: VisualizerParameters):
        self.params = params

    def draw_nodes(self, pos_matrix):
        plt.scatter(pos_matrix[:, 0], pos_matrix[:, 1], s=0.5, c=self.params.node_color)

    def visualize(self, lattice, pos_matrix, filename, override_color=None, override_line_style=None):
        """
        Creates a plot of the lattice, saved at filename

        :param pos_matrix: position matrix of each node
        :param lattice: lattice structure
        :param filename: location to save plot
        :param override_color: Only for override, default uses self.params.bond_color
        :param override_line_style: Only for override, default uses "-"
        :return (void) Saves figure at filename
        """

        # Positions of all nodes
        pos_matrix = pos_matrix.reshape((-1, 2))

        # Configure matplotlib
        plt.clf()
        plt.axis('scaled')
        plot_bounds = create_plot_bounds(pos_matrix)
        plt.xlim(plot_bounds[0], plot_bounds[1])
        plt.ylim(plot_bounds[2], plot_bounds[3])

        # Begin drawing
        if self.params.draw_nodes:
            self.draw_nodes(pos_matrix)

        if self.params.draw_bonds:
            thickness: float = 1 if lattice.get_length() < 200 else 0.3
            # Handle drawing of double lattice
            if type(lattice).__name__ == "DoubleTriangularLattice":
                draw_active_bonds(lattice.network1, pos_matrix, thickness, self.params.bond_color, "--")
                draw_active_bonds(lattice.network2, pos_matrix, thickness, "green", "-")
            else:
                color = self.params.bond_color if override_color is None else override_color
                line_style = "-" if override_line_style is None else override_line_style
                draw_active_bonds(lattice, pos_matrix, thickness, color, line_style)

        prob_fill = len(lattice.get_active_bonds()) / len(lattice.get_bonds())
        plt.title(f"p = {prob_fill:.3f}")
        # Save figure
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename)
        print("Saved visualization at", filename)

        # Create separate plot for each of glued network
        if type(lattice).__name__ == "DoubleTriangularLattice":
            file_base, file_ext = os.path.splitext(filename)
            self.visualize(lattice.network1, pos_matrix, file_base + "_network1" + file_ext,
                           override_color=self.params.bond_color,
                           override_line_style="--")
            self.visualize(lattice.network2, pos_matrix, file_base + "_network2" + file_ext,
                           override_color="green",
                           override_line_style="-")
