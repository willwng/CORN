"""
bond_histogram.py creates histograms depicting the lengths of bonds, given the position of the lattice
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import energyminimization.matrix_helper as pos
import numpy as np
import math


def create_histogram_bond_length(pos_matrix, lattice, filename):
    plt.clf()
    plt.ylabel("Frequency")
    plt.xlabel("Bond Length")
    pos_matrix = pos_matrix.reshape((-1, 2))
    edge_matrix = pos.create_edge_matrix(lattice)
    bond_length_matrix = pos.create_r_matrix(pos_matrix, edge_matrix, False)
    bonds = lattice.get_bonds()
    bond_lengths = []
    for bond in bonds:
        if bond.exists():
            node1 = bond.get_node1().get_id()
            node2 = bond.get_node2().get_id()
            length = np.linalg.norm(bond_length_matrix[node1][node2])
            bond_lengths.append(length)
    # Use Freedman–Diaconis rule to calculate the number of bins - likely causes way too many bins
    # bins = freedman_diaconis(bond_lengths)
    num_bins = 20
    # Plot actual percentages
    plt.hist(
        bond_lengths,
        bins=num_bins,
        weights=np.ones(len(bond_lengths)) / len(bond_lengths),
    )
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().set_ylim([0.0, 1.0])
    try:
        print("Saved bond histogram at", filename)
        plt.savefig(filename)
    except PermissionError:
        print("Could not save bond histogram:", filename)
    return bond_lengths


def create_histogram_bond_direction(pos_matrix, lattice, filename):
    plt.clf()
    plt.ylabel("Frequency")
    plt.xlabel("Bond Direction (Angle in Degrees)")
    pos_matrix = pos_matrix.reshape((-1, 2))
    edge_matrix = pos.create_edge_matrix(lattice)
    r_matrix = pos.create_r_matrix(pos_matrix, edge_matrix, True)
    bonds = lattice.get_bonds()
    bond_directions = []
    for bond in bonds:
        if bond.exists():
            node1 = bond.get_node1().get_id()
            node2 = bond.get_node2().get_id()
            r_vector = r_matrix[node1][node2]
            angle = math.atan2(r_vector[1], r_vector[0]) % math.pi
            angle = angle * 180.0 / math.pi
            angle = round(angle)
            bond_directions.append(angle)
    plt.hist(bond_directions)
    try:
        print("Saved bond direction histogram at", filename)
        plt.savefig(filename)
    except PermissionError:
        print("Could not save bond histogram:", filename)
    return bond_directions


def freedman_diaconis(data):
    q25, q75 = np.percentile(data, [0.25, 0.75])
    bin_width = 2 * (q75 - q25) * len(data) ** (-1 / 3)
    bins = round((max(data) - min(data)) / bin_width)
    return bins
