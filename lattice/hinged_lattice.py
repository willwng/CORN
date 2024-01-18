"""
Class to add hinges to any given lattice
"""
from lattice.abstract_lattice import AbstractLattice
from lattice.bond import Bond
from lattice.node import Node


def add_hinges(base_lattice: AbstractLattice) -> None:
    start_id = base_lattice.get_max_node_id() + 1
    node_id = start_id
    new_bonds = []
    # Add nodes between the bonds
    for bond in base_lattice.bonds:
        ((x1, y1), (x2, y2)) = bond.get_xy_xy()
        node_1, node_2 = bond.get_node1(), bond.get_node2()

        # Coordinates of new node are in the middle
        x_new, y_new = (x1 + x2) / 2, (y1 + y2) / 2
        new_node = Node(x_new, y_new, node_id)

        # Handle boundaries - (nodes that are constrained)
        if node_1.is_boundary() and node_2.is_boundary():
            new_node.set_boundary()

        # Now handle the bonds between the hinges and nodes
        bond_1 = Bond(n_1=new_node, n_2=node_1, is_present=bond.exists())
        bond_2 = Bond(n_1=new_node, n_2=node_2, is_present=bond.exists())

        # If the original bond is constrained, so are the new ones
        if bond.is_boundary():
            bond_1.set_boundary()
            bond_2.set_boundary()

        # If the bond is a periodic boundary condition, we need to set x location not to the middle
        if bond.is_hor_pbc():
            new_node.set_x(base_lattice.length + (x1 + x2 - base_lattice.length) / 2)
            if x1 > x2:
                bond_2.set_hor_pbc()  # bond to left-node is now a pbc
            else:
                bond_1.set_hor_pbc()

        new_bonds.append(bond_1)
        new_bonds.append(bond_2)
        new_node.set_hinge()
        base_lattice.nodes.append(new_node)
        node_id += 1
    base_lattice.bonds = new_bonds

    # Scale everything by 2 - so bonds are length 1
    for node in base_lattice.nodes:
        x, y = node.get_xy()
        node.set_xy(x * 2, 2 * y)
    base_lattice.set_length(base_lattice.length * 2)
    base_lattice.set_height(base_lattice.height * 2)
    print(f"Added {node_id - start_id} hinges")
    return
