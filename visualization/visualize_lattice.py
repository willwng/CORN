"""
visualize_lattice.py allows visualizations of the position of the lattice
"""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_bond(
        b,
        pos_matrix,
        # bond_length_matrix,
        lattice,
        bond_color,
        draw_edge,
        rigid_edges=None,
        largest_cluster=None,
        i_bond=0,
):
    # Line styles for various types of bonds
    thickness: float = 1 if lattice.get_length() < 200 else 0.3
    # If the bond is buckled, then the draw the line dashed
    line_style = "solid"
    # if is_bond_buckled(b, bond_length_matrix):
    #     line_style = "dashed"
    if b.is_hor_pbc():
        line_style = "dashed"
    # In the case we want to show rigid clusters (clusters w/ more than 2 bonds)
    if rigid_edges is None:
        rigid_edges = []
    if len(rigid_edges) > i_bond:
        if rigid_edges.count(rigid_edges[i_bond]) > 1 and rigid_edges[i_bond] != -1:
            bond_color = "red"
    # if in the largest cluster
    if largest_cluster is None:
        largest_cluster = []
    elif b in largest_cluster:
        bond_color = "red"

    if b.is_boundary():
        plot_node(b.get_node1(), pos_matrix, lattice, "blue", draw_edge=False)
        # bond_color = "green"
    # Get the bonded node id's
    n_1 = b.get_node1().get_id()
    n_2 = b.get_node2().get_id()
    # Normal bonds
    # colors = ["r", "g", "b"]
    if b.exists() and not b.is_hor_pbc() and not b.is_top_pbc():
        plt.plot(
            [pos_matrix[n_1][0], pos_matrix[n_2][0]],
            [pos_matrix[n_1][1], pos_matrix[n_2][1]],
            bond_color,
            linewidth=thickness,
            linestyle=line_style,
            # color=colors[b.get_direction()]
        )
    # Periodic boundary condition bonds
    if b.exists and b.is_hor_pbc() and b.is_top_pbc() and draw_edge:
        adjustment = [-lattice.get_length(), -lattice.get_height() - lattice.height_increment]
        plt.plot(
            [pos_matrix[n_1][0] + adjustment[0], pos_matrix[n_2][0]],
            [pos_matrix[n_1][1] + adjustment[1], pos_matrix[n_2][1]],
            "blue",
            linewidth=thickness,
            linestyle=line_style,
        )
        plt.plot(
            [pos_matrix[n_2][0] - adjustment[0], pos_matrix[n_1][0]],
            [pos_matrix[n_2][1] - adjustment[1], pos_matrix[n_1][1]],
            "blue",
            linewidth=thickness,
            linestyle=line_style,
        )
        return
    if b.exists() and b.is_hor_pbc() and draw_edge:
        adjustment = (
            lattice.length
            if pos_matrix[n_1][0] < pos_matrix[n_2][0]
            else -1 * lattice.length
        )
        plt.plot(
            [pos_matrix[n_1][0] + adjustment, pos_matrix[n_2][0]],
            [pos_matrix[n_1][1], pos_matrix[n_2][1]],
            "blue",
            linewidth=thickness,
            linestyle=line_style,
        )
        plt.plot(
            [pos_matrix[n_2][0] - adjustment, pos_matrix[n_1][0]],
            [pos_matrix[n_2][1], pos_matrix[n_1][1]],
            "blue",
            linewidth=thickness,
            linestyle=line_style,
        )
    if b.exists() and b.is_top_pbc() and draw_edge:
        adjustment = (
            (lattice.get_height() + lattice.height_increment)
            if pos_matrix[n_1][1] < pos_matrix[n_2][1]
            else -1 * (lattice.get_height() + lattice.height_increment)
        )
        plt.plot(
            [pos_matrix[n_1][0], pos_matrix[n_2][0]],
            [pos_matrix[n_1][1] + adjustment, pos_matrix[n_2][1]],
            "blue",
            linewidth=thickness,
            linestyle=line_style,
        )
        plt.plot(
            [pos_matrix[n_2][0], pos_matrix[n_1][0]],
            [pos_matrix[n_2][1] - adjustment, pos_matrix[n_1][1]],
            "blue",
            linewidth=thickness,
            linestyle=line_style,
        )
    return


def plot_node(n, pos_matrix, lattice, node_color, draw_edge):
    node_id = n.get_id()
    coordinates = pos_matrix[node_id]
    # plt.text(coordinates[0], coordinates[1], str(n.get_id()), color="red", fontsize=5)
    # Plot the edge nodes twice
    # if n.is_edge() and draw_edge:
    #     adjustment = (
    #         lattice.length
    #         if n.get_xy()[0] < lattice.length / 2
    #         else -1 * lattice.length
    #     )
    #     plt.scatter((coordinates[0] + adjustment), coordinates[1], s=0.5, c="blue", marker="s")
    # if n.is_top_edge() and draw_edge:
    #     adjustment = (
    #         lattice.get_height() + lattice.height_increment
    #         if n.get_xy()[1] < lattice.get_height() / 2
    #         else -1 * lattice.get_height()
    #     )
    #     plt.scatter((coordinates[0]), coordinates[1] - adjustment, s=0.5, c="orange", marker="s")
    # plt.scatter(coordinates[0], coordinates[1], s=5, c="red")
    # elif n.is_hinge():
    #     plt.scatter(coordinates[0], coordinates[1], s=0.5, c="orange")
    # else:
    #     plt.scatter(coordinates[0], coordinates[1], s=0.5, c=node_color)


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


class Visualizer:
    params: VisualizerParameters

    def __init__(self, params: VisualizerParameters):
        self.params = params

    def plot_pi_bond(self, p, pos_matrix, bond_length_matrix, lattice):
        b_1 = p.get_bond_1()
        b_2 = p.get_bond_2()
        plot_bond(
            b_1, pos_matrix, lattice, "blue", self.params.draw_edge, bond_length_matrix
        )
        plot_bond(
            b_2, pos_matrix, lattice, "blue", self.params.draw_edge, bond_length_matrix
        )
        return

    def create_plot_bounds(self, lattice):
        x_min = -2 + -1.2 * lattice.get_length() * self.params.hor_shear
        x_max = (lattice.get_length() + lattice.get_height() * self.params.hor_shear) * 1.2
        y_min = -0.1 * lattice.get_height()
        y_max = 1.1 * lattice.get_height()
        plot_bounds = (x_min, x_max, y_min, y_max)
        return plot_bounds

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
        sns.set()
        sns.set_style(style="white")  # remove background
        plt.axis("scaled")
        plot_bounds = self.create_plot_bounds(lattice)
        plt.xlim(plot_bounds[0], plot_bounds[1])
        plt.ylim(plot_bounds[2], plot_bounds[3])

        pos_matrix = pos_matrix.reshape((-1, 2))
        if self.params.draw_bonds:
            for i, bond in enumerate(lattice.get_bonds()):
                plot_bond(
                    bond,
                    pos_matrix,
                    lattice,
                    self.params.bond_color,
                    self.params.draw_edge,
                    rigid_edges=rigid_edges,
                    largest_cluster=largest_cluster,
                    i_bond=i,
                )

        if self.params.draw_nodes:
            for node in lattice.get_nodes():
                plot_node(node, pos_matrix, lattice, self.params.node_color, self.params.draw_edge)

        try:
            plt.savefig(filename)
            print("Saved visualization at", filename)
        except PermissionError:
            print("Could not save visualization:", filename)
