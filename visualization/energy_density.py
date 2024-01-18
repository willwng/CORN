"""
energy_density.py is used to create a voronoi diagram with energy density for the lattice
"""
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

from visualization.visualize_lattice import plot_bond


def add_energy_node(node, energy_arr, bond_energy):
    index = node.get_id()
    energy_arr[index] += bond_energy
    return energy_arr


def area_polygon(x, y):
    x = np.array(x)
    y = np.array(y)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_densities(vor, energies):
    densities = energies.copy()
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if -1 not in region:
            polygon_x = [vor.vertices[i][0] for i in region]
            polygon_y = [vor.vertices[i][1] for i in region]
            area = area_polygon(polygon_x, polygon_y)
            densities[r] = energies[r] / area
    return densities


def plot_voronoi(vor, densities, pos_matrix, lattice, filename):
    # Initialize plot and set bounds
    # plt.clf()
    # Normalize the colormap
    min_val = np.min(densities[np.nonzero(densities)])
    color_map = cm.get_cmap("viridis")
    norm = colors.LogNorm(vmin=max(min_val, 10 ** -20), vmax=max(densities))
    norm = colors.LogNorm(vmin=max(min_val, 10 ** -20), vmax=10 ** -5)
    mapper = cm.ScalarMappable(norm=norm, cmap=color_map)

    # Plot voronoi diagram and color based on energies value
    voronoi_plot_2d(
        vor, show_points=True, show_vertices=False, point_size=0.1, line_width=0.1
    )
    plt.show()
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if -1 not in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(densities[r]))

    # Plot the overlaying bonds
    for bond in lattice.get_bonds():
        plot_bond(bond, pos_matrix, lattice, "black", False)

    plt.axis("scaled")
    plt.title("Energy Density Voronoi Diagram")
    plt.colorbar(mapper)
    try:
        print("Saved energy density at", filename)
        plt.savefig(filename)
    except PermissionError:
        print("Could not save energy density:", filename)
    return


def create_voronoi(pos_matrix, gradients, lattice, stretch_mod, filename):
    # Get bonds and pi bonds
    bonds = lattice.get_bonds()
    active_bonds = [b for b in bonds if b.exists()]

    active_bond_indices = np.zeros((len(active_bonds), 4), dtype=np.int32)
    for i, bond in enumerate(active_bonds):
        if bond.exists():
            active_bond_indices[i][0] = bond.get_node1().get_id()
            active_bond_indices[i][1] = bond.get_node2().get_id()
            active_bond_indices[i][2] = bond.is_hor_pbc()
            active_bond_indices[i][3] = i

    # Initialize energy array
    grad_sq = np.square(gradients).reshape(-1, 2)
    grad_x, grad_y = grad_sq[:, 0], grad_sq[:, 1]
    grad_sum = grad_x + grad_y

    vor = Voronoi(pos_matrix.reshape(-1, 2))
    densities = get_densities(vor, grad_sum)
    plot_voronoi(vor, densities, pos_matrix.reshape(-1, 2), lattice, filename)
    return
