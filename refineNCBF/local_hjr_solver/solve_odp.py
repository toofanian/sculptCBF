from typing import List

import numpy as np

import hj_reachability
from refineNCBF.local_hjr_solver.breaker import BreakCriteriaChecker, MaxIterations, PostFilteredActiveSetEmpty, \
    BarrierNotMarching
from refineNCBF.local_hjr_solver.expand import SignedDistanceNeighbors, SignedDistanceNeighborsNearBoundaryDilation
from refineNCBF.local_hjr_solver.postfilter import RemoveWhereUnchanged, RemoveWhereNonNegativeHamiltonian
from refineNCBF.local_hjr_solver.prefilter import NoPreFilter, PreFilterWhereFarFromBoundarySplitOnce
from refineNCBF.local_hjr_solver.solve import LocalHjrSolver
from refineNCBF.local_hjr_solver.step_odp import DecreaseLocalHjrStepperOdp, ClassicLocalHjrStepperOdp
from refineNCBF.optimized_dp_interface.odp_dynamics import OdpDynamics
from refineNCBF.utils.types import MaskNd, ArrayNd


def create_global_solver_odp(
        dynamics: OdpDynamics,
        grid: hj_reachability.Grid,
        periodic_dims: List[int],
        avoid_set: MaskNd,
        reach_set: MaskNd,
        terminal_values: ArrayNd,

        solver_timestep: float = -0.1,
        hamiltonian_atol: float = 1e-3,
        hamiltonian_rtol: float = 1e-3,
        integration_scheme: str = 'first',
        change_fraction: float = 1,
        max_iterations: int = 100,

        verbose: bool = False,
) -> LocalHjrSolver:
    active_set_pre_filter = NoPreFilter.from_parts(
    )
    neighbor_expander = SignedDistanceNeighbors.from_parts(
        distance=np.inf
    )
    local_hjr_stepper = ClassicLocalHjrStepperOdp.from_parts(
        dynamics=dynamics,
        grid=grid,
        periodic_dims=periodic_dims,
        integration_scheme=integration_scheme,
        time_step=solver_timestep,
    )
    active_set_post_filter = RemoveWhereUnchanged.from_parts(
        atol=hamiltonian_atol,
        rtol=hamiltonian_rtol,
    )
    break_criteria_checker = BreakCriteriaChecker.from_criteria(
        [
            MaxIterations.from_parts(max_iterations=max_iterations),
            PostFilteredActiveSetEmpty.from_parts(),
            BarrierNotMarching.from_parts(change_fraction=change_fraction)
        ],
        verbose=verbose
    )

    return LocalHjrSolver(
        dynamics=dynamics,
        grid=grid,
        avoid_set=avoid_set,
        reach_set=reach_set,
        terminal_values=terminal_values,
        active_set_pre_filter=active_set_pre_filter,
        neighbor_expander=neighbor_expander,
        local_hjr_stepper=local_hjr_stepper,
        active_set_post_filter=active_set_post_filter,
        break_criteria_checker=break_criteria_checker,
        verbose=verbose,
    )


def create_decrease_global_solver_odp(
        dynamics: OdpDynamics,
        grid: hj_reachability.Grid,
        periodic_dims: List[int],
        avoid_set: MaskNd,
        reach_set: MaskNd,
        terminal_values: ArrayNd,

        solver_timestep: float = -0.1,
        hamiltonian_atol: float = 1e-3,
        hamiltonian_rtol: float = 1e-3,
        integration_scheme: str = 'first',
        change_fraction: float = 1,
        max_iterations: int = 100,

        verbose: bool = False,
) -> LocalHjrSolver:
    active_set_pre_filter = NoPreFilter.from_parts(
    )
    neighbor_expander = SignedDistanceNeighbors.from_parts(
        distance=np.inf
    )
    local_hjr_stepper = DecreaseLocalHjrStepperOdp.from_parts(
        dynamics=dynamics,
        grid=grid,
        periodic_dims=periodic_dims,
        integration_scheme=integration_scheme,
        time_step=solver_timestep,
    )
    active_set_post_filter = RemoveWhereUnchanged.from_parts(
        atol=hamiltonian_atol,
        rtol=hamiltonian_rtol,
    )
    break_criteria_checker = BreakCriteriaChecker.from_criteria(
        [
            MaxIterations.from_parts(max_iterations=max_iterations),
            PostFilteredActiveSetEmpty.from_parts(),
            BarrierNotMarching.from_parts(change_fraction=change_fraction)
        ],
        verbose=verbose
    )
    return LocalHjrSolver(
        dynamics=dynamics,
        grid=grid,
        avoid_set=avoid_set,
        reach_set=reach_set,
        terminal_values=terminal_values,
        active_set_pre_filter=active_set_pre_filter,
        neighbor_expander=neighbor_expander,
        local_hjr_stepper=local_hjr_stepper,
        active_set_post_filter=active_set_post_filter,
        break_criteria_checker=break_criteria_checker,
        verbose=verbose,
    )


def create_local_solver_odp(
        dynamics: OdpDynamics,
        grid: hj_reachability.Grid,
        periodic_dims: List[int],
        avoid_set: MaskNd,
        reach_set: MaskNd,
        terminal_values: ArrayNd,

        neighbor_distance: float = 2,
        solver_timestep: float = -0.1,
        integration_scheme: str = 'first',
        change_atol: float = 1e-3,
        change_rtol: float = 1e-3,
        max_iterations: int = 100,

        verbose: bool = False,
) -> LocalHjrSolver:
    active_set_pre_filter = NoPreFilter()
    neighbor_expander = SignedDistanceNeighbors.from_parts(
        distance=neighbor_distance
    )
    local_hjr_stepper = ClassicLocalHjrStepperOdp.from_parts(
        dynamics=dynamics,
        grid=grid,
        periodic_dims=periodic_dims,
        integration_scheme=integration_scheme,
        time_step=solver_timestep,
    )
    active_set_post_filter = RemoveWhereUnchanged.from_parts(
        atol=change_atol,
        rtol=change_rtol,
    )
    break_criteria_checker = BreakCriteriaChecker.from_criteria(
        [
            MaxIterations.from_parts(max_iterations=max_iterations),
            PostFilteredActiveSetEmpty.from_parts(),
        ],
        verbose=verbose
    )

    return LocalHjrSolver(
        dynamics=dynamics,
        grid=grid,
        avoid_set=avoid_set,
        reach_set=reach_set,
        terminal_values=terminal_values,
        active_set_pre_filter=active_set_pre_filter,
        neighbor_expander=neighbor_expander,
        local_hjr_stepper=local_hjr_stepper,
        active_set_post_filter=active_set_post_filter,
        break_criteria_checker=break_criteria_checker,
        verbose=verbose,
    )


def create_marching_solver_odp(
        dynamics: OdpDynamics,
        grid: hj_reachability.Grid,
        periodic_dims: List[int],
        avoid_set: MaskNd,
        reach_set: MaskNd,
        terminal_values: ArrayNd,

        boundary_distance_inner: int = 2,
        boundary_distance_outer: int = 2,
        neighbor_distance: int = 2,
        solver_timestep: float = -0.1,
        hamiltonian_atol: float = 1e-3,
        integration_scheme: str = 'first',
        change_fraction: float = 1,
        max_iterations: int = 100,

        verbose: bool = False,
) -> LocalHjrSolver:
    """
    NOTE: see readme for more details, info here may be inaccurate.

    classic solver with "boundary" pre-filtering, "signed distance" neighbors, "only decrease" local hjr stepper, and "no change" post-filtering.
    """
    assert solver_timestep < 0, "solver_timestep must be negative"

    # TODO: Prefilter is only relevant for the first iteration to protect against bad seed sets.
    #       Redundant with neighbor expander after first iteration.
    active_set_pre_filter = PreFilterWhereFarFromBoundarySplitOnce.from_parts(
        distance_inner=boundary_distance_inner,
        distance_outer=boundary_distance_outer,
    )
    neighbor_expander = SignedDistanceNeighborsNearBoundaryDilation.from_parts(
        neighbor_distance=neighbor_distance,
        boundary_distance_inner=boundary_distance_inner,
        boundary_distance_outer=boundary_distance_outer,
    )
    local_hjr_stepper = DecreaseLocalHjrStepperOdp.from_parts(
        dynamics=dynamics,
        grid=grid,
        periodic_dims=periodic_dims,
        integration_scheme=integration_scheme,
        time_step=solver_timestep)

    active_set_post_filter = RemoveWhereNonNegativeHamiltonian.from_parts(
        hamiltonian_atol=hamiltonian_atol
    )

    break_criteria_checker = BreakCriteriaChecker.from_criteria(
        [
            MaxIterations.from_parts(max_iterations=max_iterations),
            PostFilteredActiveSetEmpty.from_parts(),
            BarrierNotMarching.from_parts(change_fraction=change_fraction)
        ],
        verbose=verbose
    )

    return LocalHjrSolver(
        dynamics=dynamics,
        grid=grid,
        avoid_set=avoid_set,
        reach_set=reach_set,
        terminal_values=terminal_values,
        active_set_pre_filter=active_set_pre_filter,
        neighbor_expander=neighbor_expander,
        local_hjr_stepper=local_hjr_stepper,
        active_set_post_filter=active_set_post_filter,
        break_criteria_checker=break_criteria_checker,
        verbose=verbose,
    )

