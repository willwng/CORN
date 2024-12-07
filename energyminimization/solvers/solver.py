from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scipy.optimize
from scipy.sparse import spmatrix, csr_matrix

import energyminimization.energies.bend_full as bnl
import energyminimization.energies.stretch_full as snl
import energyminimization.matrix_helper as pos
from energyminimization.matrix_helper import KMatrixResult
from energyminimization.solvers.conjugate_gradient import conjugate_gradient
from energyminimization.solvers.fire import optimize_fire, optimize_fire2
from energyminimization.solvers.minimization_type import MinimizationType
from energyminimization.solvers.newton import trust_region_newton_cg
from energyminimization.transformations import Strain
from lattice.abstract_lattice import AbstractLattice


@dataclass
class ReusableResults:
    """ Values that can be reused for minimization with the same lattice and p"""
    # All Hessians for inside bonds and PBC bonds
    k_matrices_in: KMatrixResult
    k_matrices_pbc: KMatrixResult
    # Total Hessian for inside and PBC bonds
    k_matrix_in: spmatrix
    k_matrix_pbc: spmatrix
    k_matrix: spmatrix


class SolveResult:
    """ Output results from a minimization/solve """
    final_pos: np.ndarray
    individual_energies: List[float]
    init_energy: float
    final_energy: float
    info: str
    # The solver returns which results it can reuse
    reusable_results: Optional[ReusableResults]

    def __init__(self, final_pos: np.ndarray, individual_energies: List[float], init_energy: float, final_energy: float,
                 info: str, reusable_results: Optional[ReusableResults]):
        self.final_pos = final_pos
        self.individual_energies = individual_energies
        self.init_energy = init_energy
        self.final_energy = final_energy
        self.info = info
        self.reusable_results = reusable_results


@dataclass
class SolveParameters:
    """ Parameters required for minimization/solve """
    lattice: AbstractLattice
    strain: Strain
    init_pos: np.ndarray
    strained_pos: np.ndarray
    init_guess: np.ndarray
    r_matrix: np.ndarray
    correction_matrix: np.ndarray
    length_matrix: np.ndarray
    angle_matrix: np.ndarray
    active_bond_indices: np.ndarray
    active_pi_indices: np.ndarray
    stretch_mod: np.ndarray
    bend_mod: np.ndarray
    tran_mod: np.ndarray
    tolerance: float


def setup_linear_system_matrices(params: SolveParameters):
    """ Prepare the matrices for solving the linearized form of the energy """
    # Compute the K matrices (same as hessians)
    n = params.init_pos.shape[0]
    # Filter out active_bond_indices where bonds are PBC
    ind_bond_in = np.logical_and(params.active_bond_indices[:, 2] != 1, params.active_bond_indices[:, 3] != 1)
    ind_bond_pbc = np.logical_or(params.active_bond_indices[:, 2] == 1, params.active_bond_indices[:, 3] == 1)
    active_bond_indices_in = params.active_bond_indices[ind_bond_in]
    active_bond_indices_pbc = params.active_bond_indices[ind_bond_pbc]

    # Same for pi bonds
    ind_pi_in = np.logical_and(params.active_pi_indices[:, 7] != 1, params.active_pi_indices[:, 8] != 1)
    ind_pi_pbc = np.logical_or(params.active_pi_indices[:, 7] == 1, params.active_pi_indices[:, 8] == 1)
    active_pi_indices_in = params.active_pi_indices[ind_pi_in]
    active_pi_indices_pbc = params.active_pi_indices[ind_pi_pbc]

    # Filter out the moduli
    stretch_mod_in = params.stretch_mod[ind_bond_in]
    stretch_mod_pbc = params.stretch_mod[ind_bond_pbc]
    bend_mod_in = params.bend_mod[ind_pi_in]
    bend_mod_pbc = params.bend_mod[ind_pi_pbc]
    tran_mod_in = params.tran_mod[ind_bond_in]
    tran_mod_pbc = params.tran_mod[ind_bond_pbc]

    # Compute the Hessians/K matrices for inner bonds, and the PBC bonds
    k_matrices_in = pos.get_k_matrices(n=n, r_matrix=params.r_matrix, stretch_mod=stretch_mod_in,
                                       bend_mod=bend_mod_in, tran_mod=tran_mod_in,
                                       active_bond_indices=active_bond_indices_in,
                                       active_pi_indices=active_pi_indices_in)
    k_matrices_pbc = pos.get_k_matrices(n=n, r_matrix=params.r_matrix, stretch_mod=stretch_mod_pbc,
                                        bend_mod=bend_mod_pbc, tran_mod=tran_mod_pbc,
                                        active_bond_indices=active_bond_indices_pbc,
                                        active_pi_indices=active_pi_indices_pbc)

    return k_matrices_in, k_matrices_pbc, k_matrices_in.k_total, k_matrices_pbc.k_total


def linear_solve(params: SolveParameters, reusable_results: Optional[ReusableResults]):
    """
    Solves the minimization problem with linearized energy by solving up a linear system and solving
     with conjugate gradient method (optionally GPU-accelerated and/or preconditioned)
     """
    # Reuse the pre-computed matrices if possible
    if reusable_results is None:
        k_matrices_in, k_matrices_pbc, k_matrix_in, k_matrix_pbc = setup_linear_system_matrices(params=params)
        k_matrix = k_matrix_in + k_matrix_pbc
        reusable_results = ReusableResults(k_matrices_in=k_matrices_in, k_matrices_pbc=k_matrices_pbc,
                                           k_matrix_in=k_matrix_in, k_matrix_pbc=k_matrix_pbc, k_matrix=k_matrix)
    else:
        k_matrices_in = reusable_results.k_matrices_in
        k_matrices_pbc = reusable_results.k_matrices_pbc
        k_matrix_in = reusable_results.k_matrix_in
        k_matrix_pbc = reusable_results.k_matrix_pbc
        k_matrix = reusable_results.k_matrix

    # Compute the energy it takes to transform/strain the network (no relaxation)
    u_affine = pos.create_u_matrix(params.strained_pos, params.init_pos).ravel()
    # Energy from the inner bonds, PBC bonds, add for total
    _, init_energy_in = k_matrices_in.compute_quad_forms(x=u_affine)
    _, init_energy_pbc = k_matrices_pbc.compute_quad_forms(x=u_affine + params.correction_matrix)
    init_energy = init_energy_in + init_energy_pbc

    # Take advantage of initial guess if given (otherwise guess an affine displacement)
    if params.init_guess is not None:
        u_0 = pos.create_u_matrix(params.init_guess, params.init_pos).flatten()
    else:
        u_0 = u_affine
    # Initial guess to solver: we ignore the one passed in for now to reduce compounding errors
    u_0 = u_affine

    # Solve K @ u_r = b (where b is the Jacobian). Use conjugate gradients since K is likely singular
    #  (see Numerical Methods part of the paper)
    b = -(k_matrix_pbc @ params.correction_matrix.ravel())
    u_relaxed, info = conjugate_gradient(a=k_matrix, x0=u_0, b=b, tol=params.tolerance)

    if info == 1:
        print("Conjugate gradient did not converge")
        exit(1)

    # Get final position, final energies, final individual energies
    final_pos = params.init_pos + u_relaxed.reshape((-1, 2))
    final_energy_in_ind, final_energy_in = k_matrices_in.compute_quad_forms(x=u_relaxed)
    final_energy_pbd_ind, final_energy_pbc = k_matrices_pbc.compute_quad_forms(x=u_relaxed + params.correction_matrix)
    final_energy = final_energy_in + final_energy_pbc
    individual_energies = [e_in + e_pbc for e_in, e_pbc in zip(final_energy_in_ind, final_energy_pbd_ind)]

    return SolveResult(final_pos=final_pos, individual_energies=individual_energies, init_energy=init_energy,
                       final_energy=final_energy, info=str(info), reusable_results=reusable_results)


def nonlinear_solve(params: SolveParameters, minimization_type: MinimizationType):
    """
    Solves the minimization problem with linearized energy by solving up a linear system and solving
     with conjugate gradient method (optionally GPU-accelerated and/or preconditioned)
     """
    pbc = np.array([params.lattice.get_length(), params.lattice.get_height() + params.lattice.height_increment])
    correction = params.strain.apply(pos_matrix=pbc).ravel()

    # Helper function for computing the energy
    def compute_total_energy(pos_matrix: np.ndarray) -> float:
        stretch_energy = snl.get_full_stretch_energy(stretch_mod=params.stretch_mod,
                                                     pos_matrix=pos_matrix,
                                                     corrections=correction,
                                                     active_bond_indices=params.active_bond_indices,
                                                     active_bond_lengths=params.length_matrix)
        bend_energy = bnl.get_bend_energy(bend_mod=params.bend_mod,
                                          pos_matrix=pos_matrix,
                                          corrections=correction,
                                          active_pi_indices=params.active_pi_indices,
                                          orig_pi_angles=params.angle_matrix)
        return stretch_energy + bend_energy
    def compute_total_gradient(pos_matrix: np.ndarray) -> np.ndarray:
        stretch_grad = snl.get_nonlinear_stretch_jacobian(
            stretch_mod=params.stretch_mod,
            pos_matrix=pos_matrix,
            corrections=correction,
            active_bond_indices=params.active_bond_indices,
            active_bond_lengths=params.length_matrix)
        bend_grad = bnl.get_bend_jacobian(
            bend_mod=params.bend_mod,
            pos_matrix=pos_matrix,
            corrections=correction,
            active_pi_indices=params.active_pi_indices,
            orig_pi_angles=params.angle_matrix)
        return stretch_grad.ravel() + bend_grad.ravel()

    def compute_total_hessian(pos_matrix: np.ndarray) -> csr_matrix:
        stretch_hess = snl.get_nonlinear_stretch_hessian(
            stretch_mod=params.stretch_mod,
            pos_matrix=pos_matrix,
            corrections=correction,
            active_bond_indices=params.active_bond_indices,
            active_bond_lengths=params.length_matrix)
        bend_hess = bnl.get_bend_hessian(
            bend_mod=params.bend_mod,
            pos_matrix=pos_matrix,
            corrections=correction,
            active_pi_indices=params.active_pi_indices,
            orig_pi_angles=params.angle_matrix)
        return stretch_hess + bend_hess

    init_energy = compute_total_energy(pos_matrix=params.strained_pos)

    if params.init_guess is not None:
        x0 = params.init_guess.ravel()
    else:
        x0 = params.strained_pos.ravel()

    message = ""
    if minimization_type == MinimizationType.TRUST_NEWTON_CG:
        # We use the trust region Newton-CG method to solve the nonlinear problem
        final_pos, info = trust_region_newton_cg(x0=x0, fun=compute_total_energy, jac=compute_total_gradient,
                                                 hess=compute_total_hessian, g_tol=1e-6)
    elif minimization_type == MinimizationType.TRUST_CONSTR:
        max_iter = x0.size * 10
        res = scipy.optimize.minimize(fun=compute_total_energy, x0=x0, jac=compute_total_gradient,
                                      hess=compute_total_hessian, method='trust-constr',
                                      options={'gtol': 1e-6, 'maxiter': max_iter})
        final_pos = res.x
        info = int(res.status != 1 and res.status != 2) # 1 or 2 implies success
        message = res.message
    # FIRE methods
    elif minimization_type == MinimizationType.FIRE:
        final_pos, info = optimize_fire(x0=x0, df=compute_total_gradient, atol=params.tolerance)
    elif minimization_type == MinimizationType.FIRE2:
        final_pos, info = optimize_fire2(x0=x0, df=compute_total_gradient, atol=params.tolerance)
    else:
        raise ValueError(f"Unknown nonlinear solver: {minimization_type}")

    if info != 0:
        print(f"Nonlinear solver did not converge: {info}")
        print(f"Message: {message}")
        exit(1)

    final_pos = final_pos.reshape((-1, 2))
    final_energy = compute_total_energy(pos_matrix=final_pos)
    individual_energies = [final_energy, 0, 0]
    return SolveResult(final_pos=final_pos, individual_energies=individual_energies, init_energy=init_energy,
                       final_energy=final_energy, info=str(info), reusable_results=None)


def solve(params: SolveParameters, minimization_type: MinimizationType,
          reusable_results: Optional[ReusableResults]) -> SolveResult:
    """ Find the relaxed, minimal energy state of the network """
    if minimization_type == MinimizationType.LINEAR:
        return linear_solve(params=params, reusable_results=reusable_results)
    else:
        return nonlinear_solve(params=params, minimization_type=minimization_type)
