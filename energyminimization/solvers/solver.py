from enum import Enum
from typing import List, Optional, Any

import numpy as np
import scipy.linalg
from scipy.sparse import spmatrix

import energyminimization.matrix_helper as pos
from energyminimization.matrix_helper import KMatrixResult
from energyminimization.solvers.conjugate_gradient import conjugate_gradient, get_matrix_precondition, \
    hybrid_conjugate_gradient
from lattice.abstract_lattice import AbstractLattice
from visualization.energy_density import create_voronoi


class MinimizationType(Enum):
    LINEAR = 0
    LINEAR_RELAX = 1
    LINEAR_FIRE = 2
    NONLINEAR_FIRE = 3
    LINEAR_GPU = 4
    LINEAR_PRE = 5
    LINEAR_PRE_GPU = 6


class ReusableResults:
    """ Values that can be reused for minimization with the same p """
    k_matrices_in: KMatrixResult
    k_matrices_pbc: KMatrixResult
    k_matrix_in: spmatrix
    k_matrix_pbc: spmatrix
    k_matrix: spmatrix
    a_matrix: Any
    solver: Any

    def __init__(self, k_matrices_in: KMatrixResult, k_matrices_pbc: KMatrixResult, k_matrix_in: spmatrix,
                 k_matrix_pbc: spmatrix, k_matrix: spmatrix, a_matrix: Any, solver: Any):
        self.k_matrices_in = k_matrices_in
        self.k_matrices_pbc = k_matrices_pbc
        self.k_matrix_in = k_matrix_in
        self.k_matrix_pbc = k_matrix_pbc
        self.k_matrix = k_matrix
        self.a_matrix = a_matrix
        self.solver = solver


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


def linear_solve(params: SolveParameters, use_gpu: bool, use_pre: bool, reusable_results: Optional[ReusableResults]):
    """
    Solves the minimization problem with linearized energy by solving up a linear system and solving
     with conjugate gradient method (optionally GPU-accelerated and/or preconditioned)
     """
    # Reuse the pre-computed matrices if possible
    if reusable_results is None:
        k_matrices_in, k_matrices_pbc, k_matrix_in, k_matrix_pbc = setup_linear_system(params=params)
        k_matrix = k_matrix_in + k_matrix_pbc
        if use_pre:
            a_matrix, solver = get_matrix_precondition(a=k_matrix, use_gpu=use_gpu, perturb=1e-12)
        else:
            a_matrix, solver = k_matrix, None
        reusable_results = ReusableResults(k_matrices_in=k_matrices_in, k_matrices_pbc=k_matrices_pbc,
                                           k_matrix_in=k_matrix_in, k_matrix_pbc=k_matrix_pbc, k_matrix=k_matrix,
                                           a_matrix=a_matrix, solver=solver)
    else:
        k_matrices_in = reusable_results.k_matrices_in
        k_matrices_pbc = reusable_results.k_matrices_pbc
        k_matrix_in = reusable_results.k_matrix_in
        k_matrix_pbc = reusable_results.k_matrix_pbc
        k_matrix = reusable_results.k_matrix
        a_matrix, solver = reusable_results.a_matrix, reusable_results.solver

    # visualize k_matrix
    import matplotlib.pyplot as plt
    from scipy.sparse import coo_matrix

    def icholesky(a):
        n = a.shape[0]
        for k in range(n):
            a[k, k] = np.sqrt(a[k, k])
            i_, _ = a[k + 1:, k].nonzero()
            if len(i_) > 0:
                i_ = i_ + (k + 1)
                a[i_, k] = a[i_, k] / a[k, k]
            for j in i_:
                i2_, _ = a[j:n, j].nonzero()
                if len(i2_) > 0:
                    i2_ = i2_ + j
                    a[i2_, j] = a[i2_, j] - a[i2_, k] * a[j, k]

        return a

    perturb_matrix = k_matrix + 1e-12 * np.eye(k_matrix.shape[0])
    q, r = scipy.linalg.qr(perturb_matrix)
    q = coo_matrix(q)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='black')
    ax.plot(q.col, q.row, 's', color='white', ms=1)
    ax.set_xlim(0, q.shape[1])
    ax.set_ylim(0, q.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    # Count the degree of each node
    # degree = np.zeros(params.init_pos.shape[0], dtype=np.int32)
    # for i, j, _, _, _ in params.active_bond_indices:
    #     degree[i] += 1
    #     degree[j] += 1
    # degree = np.repeat(degree, 2)

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

    # Solve K @ u_r = b (where b is the Jacobian). Use conjugate gradients since K is likely singular
    b = -(k_matrix_pbc @ params.correction_matrix.ravel())

    if use_gpu and use_pre:
        u_relaxed, info = hybrid_conjugate_gradient(a_torch=a_matrix, x0=u_0, b=b, solver=solver,
                                                    tol=params.tolerance,
                                                    use_gpu=True)
    elif not use_gpu and use_pre:
        u_relaxed, info = hybrid_conjugate_gradient(a_torch=a_matrix, x0=u_0, b=b, solver=solver,
                                                    tol=params.tolerance,
                                                    use_gpu=False)
    else:
        u_relaxed, info = conjugate_gradient(a=k_matrix, x0=u_0, b=b, tol=params.tolerance, use_gpu=False)

    if info == 1:
        print("Conjugate gradient did not converge")

    final_pos = params.init_pos + u_relaxed.reshape((-1, 2))

    final_energy_in = k_matrices_in.compute_quad_forms(x=u_relaxed)[1]
    final_energy_pbc = k_matrices_pbc.compute_quad_forms(x=u_relaxed + params.correction_matrix)[1]
    # individual_energies, final_energy = k_matrices.compute_quad_forms(u_relaxed)
    final_energy = final_energy_in + final_energy_pbc
    individual_energies = [final_energy, 0, 0]

    return SolveResult(final_pos=final_pos, individual_energies=individual_energies, init_energy=init_energy,
                       final_energy=final_energy, info=str(info), reusable_results=reusable_results)


def solve(params: SolveParameters, minimization_type: MinimizationType,
          reusable_results: Optional[ReusableResults]) -> SolveResult:
    """ Find the relaxed, minimal energy state of the network """
    if minimization_type == MinimizationType.LINEAR:
        return linear_solve(params=params, use_gpu=False, use_pre=False, reusable_results=reusable_results)
    elif minimization_type == MinimizationType.LINEAR_GPU:
        return linear_solve(params=params, use_gpu=True, use_pre=False, reusable_results=reusable_results)
    elif minimization_type == MinimizationType.LINEAR_PRE:
        return linear_solve(params=params, use_gpu=False, use_pre=True, reusable_results=reusable_results)
    elif minimization_type == MinimizationType.LINEAR_PRE_GPU:
        return linear_solve(params=params, use_gpu=True, use_pre=True, reusable_results=reusable_results)
    else:
        raise ValueError(f"Unknown minimization type: {minimization_type}")
