"""
energy_heatmap.py is used to create a contour plot of energy over the region of the lattice
Currently deprecated in favor for use of energy_density.py
"""
import matplotlib.pyplot as plt
import energyminimization.matrix_helper as pos
import energyminimization.energies.stretch as stretch
import energyminimization.energies.bend as bend
import energyminimization.energies.transverse as tran
import numpy as np
import scipy.interpolate
from visualization.visualize_lattice import create_plot_bounds


def add_energy_bond_midpoint(x, y, energy, bond, bond_energy, pos_matrix):
    n1 = bond.get_node1().get_id()
    n2 = bond.get_node2().get_id()
    bond_midpoint = (pos_matrix[n1] + pos_matrix[n2]) / 2
    x_pos = bond_midpoint[0]
    y_pos = bond_midpoint[1]
    # If the bond already has been given an energy, add to it
    if (x_pos in x and y_pos in y) and x.index(x_pos) == y.index(y_pos):
        energy[x.index(x_pos)] += bond_energy
    else:
        x.append(bond_midpoint[0])
        y.append(bond_midpoint[1])
        energy.append(bond_energy)


def add_energy_vertex(x, y, energy, pi_bond, bond_energy, pos_matrix):
    vertex = pi_bond.get_vertex_node().get_id()
    x_pos = pos_matrix[vertex][0]
    y_pos = pos_matrix[vertex][1]
    # If the vertex already has been given an energy, add to it
    if (x_pos in x and y_pos in y) and x.index(x_pos) == y.index(y_pos):
        energy[x.index(x_pos)] += bond_energy
    else:
        x.append(pos_matrix[vertex][0])
        y.append(pos_matrix[vertex][1])
        energy.append(bond_energy)


def create_heatmap(pos_matrix, lattice, stretch_mod, bend_mod, tran_mod, filename):
    plt.clf()
    plt.axis("scaled")
    plot_bounds = create_plot_bounds(lattice)
    plt.xlim(plot_bounds[0], plot_bounds[1])
    plt.ylim(plot_bounds[2], plot_bounds[3])
    pos_matrix = pos_matrix.reshape((-1, 2))
    init_pos = pos.create_pos_matrix(lattice)
    edge_matrix = pos.create_edge_matrix(lattice)
    r_matrix = pos.create_r_matrix(init_pos, edge_matrix, True)
    bond_length_matrix = pos.create_r_matrix(pos_matrix, edge_matrix, False)
    u_matrix = pos.create_u_matrix(pos_matrix, init_pos)
    bonds = lattice.get_bonds()
    pi_bonds = lattice.get_pi_bonds()
    x = []
    y = []
    energy = []
    # Energy stored in the stretching of each bond
    for bond in bonds:
        if bond.exists():
            bond_energy = stretch.get_spring_energy_bond(
                bond, stretch_mod, u_matrix, r_matrix, bond_length_matrix
            )
            bond_energy += tran.get_transverse_energy_bond(
                bond, tran_mod, u_matrix, r_matrix, bond_length_matrix
            )
            add_energy_bond_midpoint(x, y, energy, bond, bond_energy, pos_matrix)

    # Energy stored in pi bond is divided evenly between two corresponding bonds
    for pi_bond in pi_bonds:
        if pi_bond.exists():
            pi_bond_energy = bend.get_bend_energy_pi_bond(
                pi_bond, bend_mod, u_matrix, r_matrix, bond_length_matrix
            )
            add_energy_vertex(x, y, energy, pi_bond, pi_bond_energy, pos_matrix)

    # Create the 2-D Grid of x,y
    x = np.array(x)
    y = np.array(y)
    xi, yi = np.linspace(0, x.max(), 200), np.linspace(0, lattice.get_height(), 200)
    xi, yi = np.meshgrid(xi, yi)
    # Interpolate missing data
    # rbf = scipy.interpolate.Rbf(x, y, energy, function='gaussian')
    # energy_interpolated = rbf(xi, yi)
    energy_interpolated = scipy.interpolate.griddata(
        (x, y), np.array(energy), (xi, yi), method="linear"
    )
    # Normalize the energies
    energy_interpolated = np.divide(energy_interpolated, np.nanmax(energy_interpolated))

    # heatmap = plt.imshow(energy_interpolated,
    #                      extent=[x.min(), x.max(), y.max(), y.min()])
    heatmap = plt.contourf(xi, yi, energy_interpolated)
    # plt.scatter(x, y)
    # for i, z in enumerate(energy):
    #     plt.annotate(round(z / max(energy), 2), (x[i], y[i]))

    plt.colorbar(heatmap)
    plt.title("Fraction of Maximum Energy Contour")
    try:
        print("Saved energy heatmap at", filename)
        plt.savefig(filename)
    except PermissionError:
        print("Could not save heatmap:", filename)
    return
