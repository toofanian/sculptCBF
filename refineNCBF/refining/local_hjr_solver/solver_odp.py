import hj_reachability
import numpy as np

from refineNCBF.refining.local_hjr_solver.breaker import BreakCriteriaChecker, MaxIterations, PostFilteredActiveSetEmpty
from refineNCBF.refining.local_hjr_solver.expand import SignedDistanceNeighbors
from refineNCBF.refining.local_hjr_solver.postfilter import RemoveWhereUnchanged
from refineNCBF.refining.local_hjr_solver.prefilter import NoPreFilter
from refineNCBF.refining.local_hjr_solver.solve import LocalHjrSolver
from refineNCBF.refining.local_hjr_solver.step_odp import ClassicLocalHjrStepperOdp
from refineNCBF.refining.optimized_dp_interface.odp_dynamics import OdpDynamics
from refineNCBF.utils.types import MaskNd, ArrayNd


def create_global_solver_odp(
        cls,
        dynamics: OdpDynamics,
        grid: hj_reachability.Grid,
        avoid_set: MaskNd,
        reach_set: MaskNd,
        terminal_values: ArrayNd,

        solver_timestep: float = -0.1,
        max_iterations: int = 100,

        verbose: bool = False,
):
    """
    NOTE: see readme for more details, info here may be inaccurate.

    classic solver with no pre-filtering, "signed distance" neighbors, "classic" local hjr stepper, and "no change" post-filtering.
    with appropriate initialization, should return the same values as vanilla/global hjr for regions connected by value to the initial active set.
    """
    active_set_pre_filter = NoPreFilter.from_parts(
    )
    neighbor_expander = SignedDistanceNeighbors.from_parts(
        distance=np.inf
    )
    local_hjr_stepper = ClassicLocalHjrStepperOdp.from_parts(
        dynamics=dynamics,
        grid=grid,
        time_step=solver_timestep,
    )
    active_set_post_filter = RemoveWhereUnchanged.from_parts(
        atol=1e-3,
        rtol=1e-3,
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
