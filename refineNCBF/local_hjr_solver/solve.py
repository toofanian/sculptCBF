import logging
import time
from typing import Callable, Union, Optional

import attr
import numpy as np

import hj_reachability
from refineNCBF.local_hjr_solver.breaker import (
    BreakCriteriaChecker,
    MaxIterations,
    PostFilteredActiveSetEmpty,
    BarrierNotMarching,
)
from refineNCBF.local_hjr_solver.expand import (
    NeighborExpander,
    SignedDistanceNeighbors,
    InnerSignedDistanceNeighbors,
    SignedDistanceNeighborsNearBoundary,
    SignedDistanceNeighborsNearBoundaryDilation,
)
from refineNCBF.local_hjr_solver.postfilter import (
    ActiveSetPostFilter,
    RemoveWhereUnchanged,
    RemoveWhereNonNegativeHamiltonian,
)
from refineNCBF.local_hjr_solver.prefilter import (
    ActiveSetPreFilter,
    NoPreFilter,
    PreFilterWhereFarFromZeroLevelset,
    PreFilterWhereOutsideZeroLevelset,
    PreFilterWhereFarFromBoundarySplit,
)
from refineNCBF.local_hjr_solver.result import LocalUpdateResult, LocalUpdateResultIteration
from refineNCBF.local_hjr_solver.step_hj import LocalHjrStepper, ClassicLocalHjrStepper, DecreaseLocalHjrStepper
from refineNCBF.local_hjr_solver.step_odp_type import OdpStepper
from refineNCBF.optimized_dp_interface.odp_dynamics import OdpDynamics
from refineNCBF.utils.types import MaskNd, ArrayNd
from refineNCBF.utils.visuals import make_configured_logger


@attr.s(auto_attribs=True)
class LocalHjrSolver(Callable):
    """
    works within this framework:

    - takes an active set and initial computed_values
    - loop:
        - pre-filters the active set
        - expands the active set to neighbors
        - performs a small-timestep local hjr solve over active set
        - post-filters the active set
        - checks break criteria
    """

    # problem setup
    _dynamics: Union[hj_reachability.Dynamics, OdpDynamics]
    _grid: hj_reachability.Grid
    _avoid_set: MaskNd
    _reach_set: MaskNd
    _terminal_values: ArrayNd

    # solver components
    _active_set_pre_filter: ActiveSetPreFilter
    _neighbor_expander: NeighborExpander
    _local_hjr_stepper: LocalHjrStepper
    _active_set_post_filter: ActiveSetPostFilter
    _break_criteria_checker: BreakCriteriaChecker

    _preloaded_result: Optional[LocalUpdateResult] = None

    _verbose: bool = False
    _logger: logging.Logger = make_configured_logger(__name__)

    def __call__(self, active_set: MaskNd, initial_values: ArrayNd) -> LocalUpdateResult:
        start_time = time.time()
        local_update_result = self._initialize_local_result(active_set, initial_values)
        while True:
            iteration = self._perform_local_update_iteration(local_update_result)
            max_diff = local_update_result.max_diff()
            cells_updated = local_update_result.get_recent_set_for_compute().sum()
            share = cells_updated / self._avoid_set.size
            blurb = f"iteration {len(local_update_result)} complete, \trunning duration is {(time.time() - start_time):.2f} seconds, \tcomputed over {cells_updated} of {self._avoid_set.size} cells ({(share * 100):.2f}%), \tmax diff: {max_diff:.2f}"
            local_update_result.add_iteration(iteration, blurb)
            if self._verbose:
                self._logger.info(blurb)
            if self._check_for_break(local_update_result):
                break
        return local_update_result

    def _initialize_local_result(self, active_set: MaskNd, initial_values: ArrayNd) -> LocalUpdateResult:
        if self._preloaded_result is None:
            return LocalUpdateResult.from_parts(
                local_solver=None if isinstance(self._local_hjr_stepper, OdpStepper) else self,
                dynamics=None if isinstance(self._local_hjr_stepper, OdpStepper) else self._dynamics,
                grid=self._grid,
                avoid_set=self._avoid_set,
                reach_set=self._reach_set,
                seed_set=active_set,
                initial_values=initial_values,
                terminal_values=self._terminal_values,
            )
        else:
            return self._preloaded_result

    def _perform_local_update_iteration(self, result: LocalUpdateResult):
        active_set_pre_filtered = self._active_set_pre_filter(result)
        active_set_expanded = self._neighbor_expander(result, active_set_pre_filtered)
        values_next = self._local_hjr_stepper(result, active_set_pre_filtered, active_set_expanded)
        active_set_post_filtered = self._active_set_post_filter(
            result, active_set_pre_filtered, active_set_expanded, values_next
        )

        return LocalUpdateResultIteration.from_parts(
            active_set_pre_filtered=active_set_pre_filtered,
            active_set_expanded=active_set_expanded,
            values_next=values_next,
            active_set_post_filtered=active_set_post_filtered,
        )

    def _check_for_break(self, result: LocalUpdateResult):
        return self._break_criteria_checker(result)

    @classmethod
    def from_parts(
        cls,
        dynamics: hj_reachability.Dynamics,
        grid: hj_reachability.Grid,
        avoid_set: MaskNd,
        reach_set: MaskNd,
        terminal_values: ArrayNd,
        prefilter: ActiveSetPreFilter,
        expander: NeighborExpander,
        stepper: LocalHjrStepper,
        postfilter: ActiveSetPostFilter,
        breaker: BreakCriteriaChecker,
        verbose: bool = False,
        solver_global_minimizing: bool = False,
    ):
        return cls(
            dynamics=dynamics,
            grid=grid,
            avoid_set=avoid_set,
            reach_set=reach_set,
            terminal_values=terminal_values,
            active_set_pre_filter=prefilter,
            neighbor_expander=expander,
            local_hjr_stepper=stepper,
            active_set_post_filter=postfilter,
            break_criteria_checker=breaker,
            verbose=verbose,
        )

    @classmethod
    def as_continue(cls, previous_result: LocalUpdateResult):
        previous_solver = LocalUpdateResult.local_solver
        previous_solver._preloaded_result = previous_result
        return previous_solver

    @classmethod
    def as_global_solver(
        cls,
        dynamics: hj_reachability.Dynamics,
        grid: hj_reachability.Grid,
        avoid_set: MaskNd,
        reach_set: MaskNd,
        terminal_values: ArrayNd,
        solver_timestep: float = -0.1,
        max_iterations: int = 100,
        change_fraction: float = 1,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        solver_accuracy=hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH,
        verbose: bool = False,
        solver_global_minimizing: bool = False,
    ):
        """
        NOTE: see readme for more details, info here may be inaccurate.

        classic solver with no pre-filtering, "signed distance" neighbors, "classic" local hjr stepper, and "no change" post-filtering.
        with appropriate initialization, should return the same values as vanilla/global hjr for regions connected by value to the initial active set.
        """
        active_set_pre_filter = NoPreFilter.from_parts()
        neighbor_expander = SignedDistanceNeighbors.from_parts(distance=np.inf)
        local_hjr_stepper = ClassicLocalHjrStepper.from_parts(
            dynamics=dynamics,
            grid=grid,
            terminal_values=terminal_values,
            time_step=solver_timestep,
            accuracy=solver_accuracy,
            verbose=verbose,
        )
        active_set_post_filter = RemoveWhereUnchanged.from_parts(
            atol=atol,
            rtol=rtol,
        )
        break_criteria_checker = BreakCriteriaChecker.from_criteria(
            [
                MaxIterations.from_parts(max_iterations=max_iterations),
                PostFilteredActiveSetEmpty.from_parts(),
                BarrierNotMarching.from_parts(change_fraction=change_fraction),
            ],
            verbose=verbose,
        )

        return cls(
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

    @classmethod
    def as_global_decrease_solver(
        cls,
        dynamics: hj_reachability.Dynamics,
        grid: hj_reachability.Grid,
        avoid_set: MaskNd,
        reach_set: MaskNd,
        terminal_values: ArrayNd,
        solver_timestep: float = -0.1,
        max_iterations: int = 100,
        change_fraction: float = 1,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        solver_accuracy=hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH,
        verbose: bool = False,
        solver_global_minimizing: bool = False,
    ):
        """
        NOTE: see readme for more details, info here may be inaccurate.

        classic solver with no pre-filtering, "signed distance" neighbors, "classic" local hjr stepper, and "no change" post-filtering.
        with appropriate initialization, should return the same values as vanilla/global hjr for regions connected by value to the initial active set.
        """
        active_set_pre_filter = NoPreFilter.from_parts()
        neighbor_expander = SignedDistanceNeighbors.from_parts(distance=np.inf)
        local_hjr_stepper = ClassicLocalHjrStepper.from_parts(
            dynamics=dynamics,
            grid=grid,
            terminal_values=terminal_values,
            time_step=solver_timestep,
            accuracy=solver_accuracy,
            verbose=verbose,
        )
        active_set_post_filter = RemoveWhereUnchanged.from_parts(
            atol=atol,
            rtol=rtol,
        )
        break_criteria_checker = BreakCriteriaChecker.from_criteria(
            [
                MaxIterations.from_parts(max_iterations=max_iterations),
                PostFilteredActiveSetEmpty.from_parts(),
                BarrierNotMarching.from_parts(change_fraction=change_fraction),
            ],
            verbose=verbose,
        )

        return cls(
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

    @classmethod
    def as_local_solver(
        cls,
        dynamics: hj_reachability.Dynamics,
        grid: hj_reachability.Grid,
        avoid_set: MaskNd,
        reach_set: MaskNd,
        terminal_values: ArrayNd,
        neighbor_distance: float = 2,
        solver_timestep: float = -0.1,
        value_change_atol: float = 1e-3,
        value_change_rtol: float = 1e-3,
        change_fraction: float = 1,
        max_iterations: int = 100,
        solver_accuracy=hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH,
        verbose: bool = False,
        solver_global_minimizing: bool = False,
    ):
        """
        NOTE: see readme for more details, info here may be inaccurate.

        classic solver with no pre-filtering, "signed distance" neighbors, "classic" local hjr stepper, and "no change" post-filtering.
        with appropriate initialization, should return the same values as vanilla/global hjr for regions connected by value to the initial active set.
        """
        active_set_pre_filter = NoPreFilter.from_parts()
        neighbor_expander = SignedDistanceNeighbors.from_parts(distance=neighbor_distance)
        local_hjr_stepper = ClassicLocalHjrStepper.from_parts(
            dynamics=dynamics,
            grid=grid,
            terminal_values=terminal_values,
            time_step=solver_timestep,
            accuracy=solver_accuracy,
            verbose=verbose,
        )
        active_set_post_filter = RemoveWhereUnchanged.from_parts(
            atol=value_change_atol,
            rtol=value_change_rtol,
        )
        break_criteria_checker = BreakCriteriaChecker.from_criteria(
            [
                MaxIterations.from_parts(max_iterations=max_iterations),
                PostFilteredActiveSetEmpty.from_parts(),
                BarrierNotMarching.from_parts(change_fraction=change_fraction),
            ],
            verbose=verbose,
        )

        return cls(
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

    @classmethod
    def as_marching_solver(
        cls,
        dynamics: hj_reachability.Dynamics,
        grid: hj_reachability.Grid,
        avoid_set: MaskNd,
        reach_set: MaskNd,
        terminal_values: ArrayNd,
        boundary_distance_inner: float = 2,
        boundary_distance_outer: float = 2,
        neighbor_distance: float = 2,
        solver_timestep: float = -0.1,
        hamiltonian_atol: float = 1e-3,
        change_fraction: float = 1,
        max_iterations: int = 100,
        solver_accuracy=hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH,
        verbose: bool = False,
        solver_global_minimizing: bool = False,
    ):
        """
        NOTE: see readme for more details, info here may be inaccurate.

        classic solver with "boundary" pre-filtering, "signed distance" neighbors, "only decrease" local hjr stepper, and "no change" post-filtering.
        """
        assert solver_timestep < 0, "solver_timestep must be negative"

        # TODO: Prefilter is only relevant for the first iteration to protect against bad seed sets.
        #       Redundant with neighbor expander after first iteration.
        active_set_pre_filter = PreFilterWhereFarFromBoundarySplit.from_parts(
            distance_inner=boundary_distance_inner,
            distance_outer=boundary_distance_outer,
        )
        neighbor_expander = SignedDistanceNeighborsNearBoundaryDilation.from_parts(
            neighbor_distance=neighbor_distance,
            boundary_distance_inner=boundary_distance_inner,
            boundary_distance_outer=boundary_distance_outer,
        )
        local_hjr_stepper = DecreaseLocalHjrStepper.from_parts(
            dynamics=dynamics,
            grid=grid,
            terminal_values=terminal_values,
            time_step=solver_timestep,
            accuracy=solver_accuracy,
            verbose=verbose,
        )
        active_set_post_filter = RemoveWhereNonNegativeHamiltonian.from_parts(hamiltonian_atol=hamiltonian_atol)
        break_criteria_checker = BreakCriteriaChecker.from_criteria(
            [
                MaxIterations.from_parts(max_iterations=max_iterations),
                PostFilteredActiveSetEmpty.from_parts(),
                BarrierNotMarching.from_parts(change_fraction=change_fraction),
            ],
            verbose=verbose,
        )

        return cls(
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

    @classmethod
    def as_decrease(
        cls,
        dynamics: hj_reachability.Dynamics,
        grid: hj_reachability.Grid,
        avoid_set: MaskNd,
        reach_set: MaskNd,
        terminal_values: ArrayNd,
        neighbor_distance: float = 1,
        solver_timestep: float = -0.1,
        max_iterations: int = 100,
        verbose: bool = False,
        solver_global_minimizing: bool = False,
    ):
        """
        NOTE: see readme for more details, info here may be inaccurate.

        classic solver with no pre-filtering, signed distance neighbors, only decrease local hjr stepper, and no change post-filtering.
        with appropriate initialization, should return the same zero levelset as vanilla/global hjr for regions connected by value to the initial active set.
        values should be conservative (low) generally.
        """
        active_set_pre_filter = NoPreFilter.from_parts()
        neighbor_expander = SignedDistanceNeighbors.from_parts(distance=neighbor_distance)
        local_hjr_stepper = DecreaseLocalHjrStepper.from_parts(
            dynamics=dynamics, grid=grid, terminal_values=terminal_values, time_step=solver_timestep, verbose=verbose
        )
        active_set_post_filter = RemoveWhereNonNegativeHamiltonian.from_parts()
        break_criteria_checker = BreakCriteriaChecker.from_criteria(
            [
                MaxIterations.from_parts(max_iterations=max_iterations),
                PostFilteredActiveSetEmpty.from_parts(),
            ],
            verbose=verbose,
        )

        return cls(
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

    @classmethod
    def as_boundary_solver(
        cls,
        dynamics: hj_reachability.Dynamics,
        grid: hj_reachability.Grid,
        avoid_set: MaskNd,
        reach_set: MaskNd,
        terminal_values: ArrayNd,
        boundary_distance: float = 1,
        neighbor_distance: float = 1,
        solver_timestep: float = -0.1,
        value_change_atol: float = 1e-3,
        value_change_rtol: float = 1e-3,
        max_iterations: int = 100,
        verbose: bool = False,
        solver_global_minimizing: bool = False,
    ):
        """
        NOTE: see readme for more details, info here may be inaccurate.

        classic solver with "boundary" pre-filtering, "signed distance" neighbors, "classic" local hjr stepper, and "no change" post-filtering.
        """
        active_set_pre_filter = PreFilterWhereFarFromZeroLevelset.from_parts(distance=boundary_distance)
        neighbor_expander = SignedDistanceNeighbors.from_parts(distance=neighbor_distance)
        local_hjr_stepper = ClassicLocalHjrStepper.from_parts(
            dynamics=dynamics, grid=grid, terminal_values=terminal_values, time_step=solver_timestep, verbose=verbose
        )
        active_set_post_filter = RemoveWhereUnchanged.from_parts(
            atol=value_change_atol,
            rtol=value_change_rtol,
        )
        break_criteria_checker = BreakCriteriaChecker.from_criteria(
            [
                MaxIterations.from_parts(max_iterations=max_iterations),
                PostFilteredActiveSetEmpty.from_parts(),
            ],
            verbose=verbose,
        )

        return cls(
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

    @classmethod
    def as_boundary_decrease(
        cls,
        dynamics: hj_reachability.Dynamics,
        grid: hj_reachability.Grid,
        avoid_set: MaskNd,
        reach_set: MaskNd,
        terminal_values: ArrayNd,
        boundary_distance: float = 1,
        neighbor_distance: float = 1,
        solver_timestep: float = -0.1,
        max_iterations: int = 100,
        verbose: bool = False,
        solver_global_minimizing: bool = False,
    ):
        """
        NOTE: see readme for more details, info here may be inaccurate.

        classic solver with "boundary" pre-filtering, "signed distance" neighbors, "only decrease" local hjr stepper, and "no change" post-filtering.
        """
        active_set_pre_filter = PreFilterWhereFarFromZeroLevelset.from_parts(distance=boundary_distance)
        neighbor_expander = SignedDistanceNeighbors.from_parts(distance=neighbor_distance)
        local_hjr_stepper = DecreaseLocalHjrStepper.from_parts(
            dynamics=dynamics, grid=grid, terminal_values=terminal_values, time_step=solver_timestep, verbose=verbose
        )
        active_set_post_filter = RemoveWhereNonNegativeHamiltonian.from_parts()
        break_criteria_checker = BreakCriteriaChecker.from_criteria(
            [
                MaxIterations.from_parts(max_iterations=max_iterations),
                PostFilteredActiveSetEmpty.from_parts(),
            ],
            verbose=verbose,
        )

        return cls(
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

    @classmethod
    def as_inner_decrease(
        cls,
        dynamics: hj_reachability.Dynamics,
        grid: hj_reachability.Grid,
        avoid_set: MaskNd,
        reach_set: MaskNd,
        terminal_values: ArrayNd,
        boundary_distance: float = 1,
        neighbor_distance: float = 1,
        solver_timestep: float = -0.1,
        value_change_atol: float = 1e-3,
        value_change_rtol: float = 1e-3,
        max_iterations: int = 100,
        verbose: bool = False,
        solver_global_minimizing: bool = False,
    ):
        """
        NOTE: see readme for more details, info here may be inaccurate.

        classic solver with "boundary" pre-filtering, "signed distance" neighbors, "only decrease" local hjr stepper, and "no change" post-filtering.
        """
        active_set_pre_filter = PreFilterWhereOutsideZeroLevelset.from_parts()
        neighbor_expander = InnerSignedDistanceNeighbors.from_parts(distance=neighbor_distance)
        local_hjr_stepper = DecreaseLocalHjrStepper.from_parts(
            dynamics=dynamics, grid=grid, terminal_values=terminal_values, time_step=solver_timestep, verbose=verbose
        )
        active_set_post_filter = RemoveWhereUnchanged.from_parts(
            atol=value_change_atol,
            rtol=value_change_rtol,
        )
        break_criteria_checker = BreakCriteriaChecker.from_criteria(
            [
                MaxIterations.from_parts(max_iterations=max_iterations),
                PostFilteredActiveSetEmpty.from_parts(),
            ],
            verbose=verbose,
        )

        return cls(
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

    @classmethod
    def as_boundary_decrease_split(
        cls,
        dynamics: hj_reachability.Dynamics,
        grid: hj_reachability.Grid,
        avoid_set: MaskNd,
        reach_set: MaskNd,
        terminal_values: ArrayNd,
        boundary_distance_inner: float = 1,
        boundary_distance_outer: float = 1,
        neighbor_distance: float = 1,
        solver_timestep: float = -0.1,
        max_iterations: int = 100,
        verbose: bool = False,
        solver_global_minimizing: bool = False,
    ):
        """
        NOTE: see readme for more details, info here may be inaccurate.

        classic solver with "boundary" pre-filtering, "signed distance" neighbors, "only decrease" local hjr stepper, and "no change" post-filtering.
        """
        assert solver_timestep < 0, "solver_timestep must be negative"

        active_set_pre_filter = NoPreFilter.from_parts()
        neighbor_expander = SignedDistanceNeighborsNearBoundary.from_parts(
            neighbor_distance=neighbor_distance,
            boundary_distance_inner=boundary_distance_inner,
            boundary_distance_outer=boundary_distance_outer,
        )
        local_hjr_stepper = DecreaseLocalHjrStepper.from_parts(
            dynamics=dynamics, grid=grid, terminal_values=terminal_values, time_step=solver_timestep, verbose=verbose
        )
        active_set_post_filter = RemoveWhereNonNegativeHamiltonian.from_parts()
        break_criteria_checker = BreakCriteriaChecker.from_criteria(
            [
                MaxIterations.from_parts(max_iterations=max_iterations),
                PostFilteredActiveSetEmpty.from_parts(),
            ],
            verbose=verbose,
        )

        return cls(
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

    @classmethod
    def as_custom(
        cls,
        dynamics: hj_reachability.Dynamics,
        grid: hj_reachability.Grid,
        avoid_set: MaskNd,
        reach_set: MaskNd,
        terminal_values: ArrayNd,
        boundary_distance_inner: float = 1,
        boundary_distance_outer: float = 1,
        neighbor_distance: float = 1,
        solver_timestep: float = -0.1,
        max_iterations: int = 100,
        verbose: bool = False,
        solver_global_minimizing: bool = False,
    ):
        assert solver_timestep < 0, "solver_timestep must be negative"

        active_set_pre_filter = NoPreFilter.from_parts()
        neighbor_expander = SignedDistanceNeighborsNearBoundary.from_parts(
            neighbor_distance=neighbor_distance,
            boundary_distance_inner=np.inf,
            boundary_distance_outer=6,
        )
        local_hjr_stepper = ClassicLocalHjrStepper.from_parts(
            dynamics=dynamics, grid=grid, terminal_values=terminal_values, time_step=solver_timestep, verbose=verbose
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
            verbose=verbose,
        )

        return cls(
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
