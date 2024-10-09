from typing import List, Optional

import numpy as np
from scipy.sparse import spmatrix, csr_matrix

import energyminimization.energies.stretch_nonlinear as snl
import energyminimization.matrix_helper as pos
from energyminimization.matrix_helper import KMatrixResult
from energyminimization.solvers.conjugate_gradient import conjugate_gradient
from energyminimization.solvers.minimization_type import MinimizationType
from energyminimization.solvers.newton import trust_region_newton_cg
from lattice.abstract_lattice import AbstractLattice


class ReusableResults:
    """ Values that can be reused for minimization with the same lattice and p"""
    # All Hessians for inside bonds and PBC bonds
    k_matrices_in: KMatrixResult
    k_matrices_pbc: KMatrixResult
    # Total Hessian for inside and PBC bonds
    k_matrix_in: spmatrix
    k_matrix_pbc: spmatrix
    k_matrix: spmatrix

    def __init__(self, k_matrices_in: KMatrixResult, k_matrices_pbc: KMatrixResult, k_matrix_in: spmatrix,
                 k_matrix_pbc: spmatrix, k_matrix: spmatrix):
        self.k_matrices_in = k_matrices_in
        self.k_matrices_pbc = k_matrices_pbc
        self.k_matrix_in = k_matrix_in
        self.k_matrix_pbc = k_matrix_pbc
        self.k_matrix = k_matrix


class SolveResult:
    """ Output results from a minimization/solve """
    final_pos: np.ndarray
    individual_energies: List[float]
    init_energy: float
    final_energy: float
    info: str
    reusable_results: Optional[ReusableResults]

    def __init__(self, final_pos: np.ndarray, individual_energies: List[float], init_energy: float, final_energy: float,
                 info: str, reusable_results: Optional[ReusableResults]):
        self.final_pos = final_pos
        self.individual_energies = individual_energies
        self.init_energy = init_energy
        self.final_energy = final_energy
        self.info = info
        self.reusable_results = reusable_results


class SolveParameters:
    """ Parameters required for minimization/solve """
    lattice: AbstractLattice
    init_pos: np.ndarray
    sheared_pos: np.ndarray
    init_guess: np.ndarray
    r_matrix: np.ndarray
    correction_matrix: np.ndarray
    length_matrix: np.ndarray
    active_bond_indices: np.ndarray
    active_pi_indices: np.ndarray
    stretch_mod: float
    bend_mod: float
    tran_mod: float
    tolerance: float

    def __init__(self, lattice: AbstractLattice, init_pos: np.ndarray, sheared_pos: np.ndarray, init_guess: np.ndarray,
                 r_matrix: np.ndarray, correction_matrix: np.ndarray, length_matrix: np.ndarray,
                 active_bond_indices: np.ndarray, active_pi_indices: np.ndarray, stretch_mod: float, bend_mod: float,
                 tran_mod: float, tolerance: float):
        self.lattice = lattice
        self.init_pos = init_pos
        self.sheared_pos = sheared_pos
        self.init_guess = init_guess
        self.r_matrix = r_matrix
        self.correction_matrix = correction_matrix
        self.length_matrix = length_matrix
        self.active_bond_indices = active_bond_indices
        self.active_pi_indices = active_pi_indices
        self.stretch_mod = stretch_mod
        self.bend_mod = bend_mod
        self.tran_mod = tran_mod
        self.tolerance = tolerance


def setup_linear_system(params: SolveParameters):
    """ Prepare the matrices for solving the linearized form of the energy """
    # Compute the K matrices (same as hessians)
    n = params.init_pos.shape[0]
    # Filter out active_bond_indices where bonds are PBC
    active_bond_indices_in = params.active_bond_indices[
        np.logical_and(params.active_bond_indices[:, 2] != 1, params.active_bond_indices[:, 3] != 1)]
    active_bond_indices_pbc = params.active_bond_indices[
        np.logical_or(params.active_bond_indices[:, 2] == 1, params.active_bond_indices[:, 3] == 1)]

    k_matrices_in = pos.get_k_matrices(n=n, r_matrix=params.r_matrix, stretch_mod=params.stretch_mod,
                                       bend_mod=params.bend_mod, tran_mod=params.tran_mod,
                                       active_bond_indices=active_bond_indices_in,
                                       active_pi_indices=params.active_pi_indices)

    k_matrices_pbc = pos.get_k_matrices(n=n, r_matrix=params.r_matrix, stretch_mod=params.stretch_mod,
                                        bend_mod=params.bend_mod, tran_mod=params.tran_mod,
                                        active_bond_indices=active_bond_indices_pbc,
                                        active_pi_indices=params.active_pi_indices)

    return k_matrices_in, k_matrices_pbc, k_matrices_in.k_total, k_matrices_pbc.k_total


def linear_solve(params: SolveParameters, reusable_results: Optional[ReusableResults]):
    """
    Solves the minimization problem with linearized energy by solving up a linear system and solving
     with conjugate gradient method (optionally GPU-accelerated and/or preconditioned)
     """
    # Reuse the pre-computed matrices if possible
    if reusable_results is None:
        k_matrices_in, k_matrices_pbc, k_matrix_in, k_matrix_pbc = setup_linear_system(params=params)
        k_matrix = k_matrix_in + k_matrix_pbc
        reusable_results = ReusableResults(k_matrices_in=k_matrices_in, k_matrices_pbc=k_matrices_pbc,
                                           k_matrix_in=k_matrix_in, k_matrix_pbc=k_matrix_pbc, k_matrix=k_matrix)
    else:
        k_matrices_in = reusable_results.k_matrices_in
        k_matrices_pbc = reusable_results.k_matrices_pbc
        k_matrix_in = reusable_results.k_matrix_in
        k_matrix_pbc = reusable_results.k_matrix_pbc
        k_matrix = reusable_results.k_matrix

    # Compute the energy it takes to transform the network without relaxation
    u_affine = pos.create_u_matrix(params.sheared_pos, params.init_pos).ravel()
    init_energy_in = k_matrices_in.compute_quad_forms(x=u_affine)[1]
    init_energy_pbc = k_matrices_pbc.compute_quad_forms(x=u_affine + params.correction_matrix)[1]
    init_energy = init_energy_in + init_energy_pbc

    # Take advantage of initial guess if given (otherwise guess an affine displacement)
    if params.init_guess is not None:
        u_0 = pos.create_u_matrix(params.init_guess, params.init_pos).flatten()
    else:
        u_0 = u_affine

    # Ignore initial guess for now, to reduce possible compounding errors
    u_0 = u_affine

    # Solve K @ u_r = b (where b is the Jacobian). Use conjugate gradients since K is likely singular
    b = -(k_matrix_pbc @ params.correction_matrix.ravel())

    u_relaxed, info = conjugate_gradient(a=k_matrix, x0=u_0, b=b, tol=params.tolerance)

    if info == 1:
        print("Conjugate gradient did not converge")
        exit(1)

    final_pos = params.init_pos + u_relaxed.reshape((-1, 2))

    final_energy_in = k_matrices_in.compute_quad_forms(x=u_relaxed)[1]
    final_energy_pbc = k_matrices_pbc.compute_quad_forms(x=u_relaxed + params.correction_matrix)[1]
    # individual_energies, final_energy = k_matrices.compute_quad_forms(u_relaxed)
    final_energy = final_energy_in + final_energy_pbc
    individual_energies = [final_energy, 0, 0]

    return SolveResult(final_pos=final_pos, individual_energies=individual_energies, init_energy=init_energy,
                       final_energy=final_energy, info=str(info), reusable_results=reusable_results)


def nonlinear_solve(params: SolveParameters):
    """
    Solves the minimization problem with linearized energy by solving up a linear system and solving
     with conjugate gradient method (optionally GPU-accelerated and/or preconditioned)
     """
    u_affine = pos.create_u_matrix(params.sheared_pos, params.init_pos).ravel()

    # Sort the bonds into inner degrees of freedom and periodic boundary conditions
    bond_indices = params.active_bond_indices
    active_bond_indices_in = bond_indices[(bond_indices[:, 2] == 0) & (bond_indices[:, 3] == 0), :]
    active_bond_indices_pbc = bond_indices[(bond_indices[:, 2] == 1) | (bond_indices[:, 3] == 1), :]

    # Helper function for computing the energy
    def compute_energy(u: np.ndarray, active_bond_indices: np.ndarray) -> float:
        return snl.get_nonlinear_stretch_energy(stretch_mod=params.stretch_mod,
                                                u_node_matrix=u,
                                                r_matrix=params.r_matrix,
                                                active_bond_indices=active_bond_indices,
                                                active_bond_lengths=params.length_matrix)

    def compute_total_energy(u) -> float:
        init_energy_in = compute_energy(u=u, active_bond_indices=active_bond_indices_in)
        init_energy_pbc = compute_energy(u=u + params.correction_matrix, active_bond_indices=active_bond_indices_pbc)
        return init_energy_in + init_energy_pbc

    def compute_gradient(u: np.ndarray, active_bond_indices: np.ndarray) -> np.ndarray:
        return snl.get_nonlinear_stretch_jacobian(
            stretch_mod=params.stretch_mod,
            u_node_matrix=u,
            r_matrix=params.r_matrix,
            active_bond_indices=active_bond_indices,
            active_bond_lengths=params.length_matrix)

    def compute_total_gradient(u: np.ndarray) -> np.ndarray:
        gradient_in = compute_gradient(u=u, active_bond_indices=active_bond_indices_in)
        gradient_pbc = compute_gradient(u=u + params.correction_matrix, active_bond_indices=active_bond_indices_pbc)
        return gradient_in + gradient_pbc

    def compute_hessian(u: np.ndarray, active_bond_indices: np.ndarray) -> csr_matrix:
        return snl.get_nonlinear_stretch_hessian(
            stretch_mod=params.stretch_mod,
            u_node_matrix=u,
            r_matrix=params.r_matrix,
            active_bond_indices=active_bond_indices,
            active_bond_lengths=params.length_matrix)

    def compute_total_hessian(u: np.ndarray) -> csr_matrix:
        hessian_in = compute_hessian(u=u, active_bond_indices=active_bond_indices_in)
        hessian_pbc = compute_hessian(u=u + params.correction_matrix, active_bond_indices=active_bond_indices_pbc)
        return hessian_in + hessian_pbc

    init_energy = compute_total_energy(u=u_affine)

    if params.init_guess is not None:
        u_0 = pos.create_u_matrix(params.init_guess, params.init_pos).flatten()
    else:
        u_0 = u_affine
    u_relaxed, info = trust_region_newton_cg(x0=u_0, fun=compute_total_energy, jac=compute_total_gradient,
                                             hess=compute_total_hessian, g_tol=1e-5)

    final_pos = params.init_pos + u_relaxed.reshape((-1, 2))
    final_energy = compute_total_energy(u=u_relaxed)
    individual_energies = [final_energy, 0, 0]
    return SolveResult(final_pos=final_pos, individual_energies=individual_energies, init_energy=init_energy,
                       final_energy=final_energy, info=str(info), reusable_results=None)


def solve(params: SolveParameters, minimization_type: MinimizationType,
          reusable_results: Optional[ReusableResults]) -> SolveResult:
    """ Find the relaxed, minimal energy state of the network """
    if minimization_type == MinimizationType.LINEAR:
        return linear_solve(params=params, reusable_results=reusable_results)
    if minimization_type == MinimizationType.NONLINEAR:
        return nonlinear_solve(params=params)
    else:
        raise ValueError(f"Unknown minimization type: {minimization_type}")
