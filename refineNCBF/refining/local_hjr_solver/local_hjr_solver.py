from typing import Callable

import attr
import hj_reachability

from refineNCBF.refining.hj_reachability_interface.hj_setup import HjSetup
from refineNCBF.refining.local_hjr_solver.active_set_post_filter import ActiveSetPostFilter, RemoveWhereUnchanged
from refineNCBF.refining.local_hjr_solver.active_set_pre_filter import ActiveSetPreFilter, NoFilter, FilterWhereFarFromZeroLevelset
from refineNCBF.refining.local_hjr_solver.break_criteria_checker import BreakCriteriaChecker, MaxIterations, PostFilteredActiveSetEmpty
from refineNCBF.refining.local_hjr_solver.local_hjr_result import LocalUpdateResult, LocalUpdateResultIteration
from refineNCBF.refining.local_hjr_solver.local_hjr_stepper import LocalHjrStepper, ClassicLocalHjrStepper, OnlyDecreaseLocalHjrStepper
from refineNCBF.refining.local_hjr_solver.neighbor_expander import NeighborExpander, SignedDistanceNeighbors
from refineNCBF.utils.types import MaskNd, ArrayNd


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
    _hj_setup: HjSetup
    _solver_settings: hj_reachability.SolverSettings
    _avoid_set: MaskNd
    _reach_set: MaskNd

    # solver components
    _active_set_pre_filter: ActiveSetPreFilter
    _neighbor_expander: NeighborExpander
    _local_hjr_stepper: LocalHjrStepper
    _active_set_post_filter: ActiveSetPostFilter
    _break_criteria_checker: BreakCriteriaChecker

    _verbose: bool = False

    def __call__(self, active_set: MaskNd, initial_values: ArrayNd) -> LocalUpdateResult:
        local_update_result = self._initialize_local_result(active_set, initial_values)
        while True:
            iteration = self._perform_local_update_iteration(local_update_result)
            local_update_result.add_iteration(iteration)
            if self._check_for_break(local_update_result):
                break
        return local_update_result

    def _initialize_local_result(self, active_set: MaskNd, initial_values: ArrayNd) -> LocalUpdateResult:
        return LocalUpdateResult.from_parts(
            hj_setup=self._hj_setup,
            avoid_set=self._avoid_set,
            reach_set=self._reach_set,
            seed_set=active_set,
            initial_values=initial_values,
        )

    def _perform_local_update_iteration(self, result: LocalUpdateResult):
        active_set_pre_filtered = self._active_set_pre_filter(
            result
        )
        active_set_expanded = self._neighbor_expander(
            result, active_set_pre_filtered
        )
        if self._verbose:
            print(f'computing hamiltonian over {active_set_expanded.sum()} points')
        values_next = self._local_hjr_stepper(
            result, active_set_pre_filtered, active_set_expanded
        )
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
            hj_setup: HjSetup,
            solver_settings: hj_reachability.SolverSettings,
            avoid_set: MaskNd,
            reach_set: MaskNd,
            active_set_pre_filter: ActiveSetPreFilter,
            neighbor_expander: NeighborExpander,
            local_hjr_stepper: LocalHjrStepper,
            active_set_post_filter: ActiveSetPostFilter,
            break_criteria_checker: BreakCriteriaChecker,
            verbose: bool = False,
    ):
        return cls(
            hj_setup=hj_setup,
            solver_settings=solver_settings,
            avoid_set=avoid_set,
            reach_set=reach_set,
            active_set_pre_filter=active_set_pre_filter,
            neighbor_expander=neighbor_expander,
            local_hjr_stepper=local_hjr_stepper,
            active_set_post_filter=active_set_post_filter,
            break_criteria_checker=break_criteria_checker,
            verbose=verbose,
        )

    @classmethod
    def as_classic_solver(
            cls,
            hj_setup: HjSetup,
            solver_settings: hj_reachability.SolverSettings,
            avoid_set: MaskNd,
            reach_set: MaskNd,
            neighbor_distance: float = 1.0,
            solver_timestep: float = -0.1,
            value_change_atol: float = 1e-3,
            value_change_rtol: float = 1e-3,
            max_iterations: int = 100,
            verbose: bool = False,
    ):
        """
        NOTE: see readme for more details, info here may be inaccurate.

        classic solver with no pre-filtering, "signed distance" neighbors, "classic" local hjr stepper, and "no change" post-filtering.
        with appropriate initialization, should return the same values as vanilla/global hjr for regions connected by value to the initial active set.
        """
        active_set_pre_filter = NoFilter.from_parts(
        )
        neighbor_expander = SignedDistanceNeighbors.from_parts(
            distance=neighbor_distance
        )
        local_hjr_stepper = ClassicLocalHjrStepper.from_parts(
            hj_setup=hj_setup,
            solver_settings=solver_settings,
            time_step=solver_timestep,
            verbose=verbose
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
            verbose=verbose
        )

        return cls(
            hj_setup=hj_setup,
            solver_settings=solver_settings,
            avoid_set=avoid_set,
            reach_set=reach_set,
            active_set_pre_filter=active_set_pre_filter,
            neighbor_expander=neighbor_expander,
            local_hjr_stepper=local_hjr_stepper,
            active_set_post_filter=active_set_post_filter,
            break_criteria_checker=break_criteria_checker,
            verbose=verbose,
        )

    @classmethod
    def as_only_decrease(
            cls,
            hj_setup: HjSetup,
            solver_settings: hj_reachability.SolverSettings,
            avoid_set: MaskNd,
            reach_set: MaskNd,

            boundary_distance: float = 1.0,
            neighbor_distance: float = 1.0,
            solver_timestep: float = -0.1,
            value_change_atol: float = 1e-3,
            value_change_rtol: float = 1e-3,
            max_iterations: int = 100,

            verbose: bool = False,
    ):
        """
        NOTE: see readme for more details, info here may be inaccurate.

        classic solver with no pre-filtering, signed distance neighbors, only decrease local hjr stepper, and no change post-filtering.
        with appropriate initialization, should return the same zero levelset as vanilla/global hjr for regions connected by value to the initial active set.
        values should be conservative (low) generally.
        """
        active_set_pre_filter = FilterWhereFarFromZeroLevelset.from_parts(
            distance=boundary_distance
        )
        neighbor_expander = SignedDistanceNeighbors.from_parts(
            distance=neighbor_distance
        )
        local_hjr_stepper = ClassicLocalHjrStepper.from_parts(
            hj_setup=hj_setup,
            solver_settings=solver_settings,
            time_step=solver_timestep,
            verbose=verbose
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
            verbose=verbose
        )

        return cls(
            hj_setup=hj_setup,
            solver_settings=solver_settings,
            avoid_set=avoid_set,
            reach_set=reach_set,
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
            hj_setup: HjSetup,
            solver_settings: hj_reachability.SolverSettings,
            avoid_set: MaskNd,
            reach_set: MaskNd,

            boundary_distance: float = 1.0,
            neighbor_distance: float = 1.0,
            solver_timestep: float = -0.1,
            value_change_atol: float = 1e-3,
            value_change_rtol: float = 1e-3,
            max_iterations: int = 100,

            verbose: bool = False,
    ):
        """
        NOTE: see readme for more details, info here may be inaccurate.

        classic solver with "boundary" pre-filtering, "signed distance" neighbors, "classic" local hjr stepper, and "no change" post-filtering.
        """
        active_set_pre_filter = FilterWhereFarFromZeroLevelset.from_parts(
            distance=boundary_distance
        )
        neighbor_expander = SignedDistanceNeighbors.from_parts(
            distance=neighbor_distance
        )
        local_hjr_stepper = ClassicLocalHjrStepper.from_parts(
            hj_setup=hj_setup,
            solver_settings=solver_settings,
            time_step=solver_timestep,
            verbose=verbose
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
            verbose=verbose
        )

        return cls(
            hj_setup=hj_setup,
            solver_settings=solver_settings,
            avoid_set=avoid_set,
            reach_set=reach_set,
            active_set_pre_filter=active_set_pre_filter,
            neighbor_expander=neighbor_expander,
            local_hjr_stepper=local_hjr_stepper,
            active_set_post_filter=active_set_post_filter,
            break_criteria_checker=break_criteria_checker,
            verbose=verbose,
        )

    @classmethod
    def as_boundary_solver_with_only_decrease(
            cls,
            hj_setup: HjSetup,
            solver_settings: hj_reachability.SolverSettings,
            avoid_set: MaskNd,
            reach_set: MaskNd,

            boundary_distance: float = 1.0,
            neighbor_distance: float = 1.0,
            solver_timestep: float = -0.1,
            value_change_atol: float = 1e-3,
            value_change_rtol: float = 1e-3,
            max_iterations: int = 100,

            verbose: bool = False,
    ):
        """
        NOTE: see readme for more details, info here may be inaccurate.

        classic solver with "boundary" pre-filtering, "signed distance" neighbors, "only decrease" local hjr stepper, and "no change" post-filtering.
        """
        active_set_pre_filter = FilterWhereFarFromZeroLevelset.from_parts(
            distance=boundary_distance
        )
        neighbor_expander = SignedDistanceNeighbors.from_parts(
            distance=neighbor_distance
        )
        local_hjr_stepper = OnlyDecreaseLocalHjrStepper.from_parts(
            hj_setup=hj_setup,
            solver_settings=solver_settings,
            time_step=solver_timestep,
            verbose=verbose
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
            verbose=verbose
        )

        return cls(
            hj_setup=hj_setup,
            solver_settings=solver_settings,
            avoid_set=avoid_set,
            reach_set=reach_set,
            active_set_pre_filter=active_set_pre_filter,
            neighbor_expander=neighbor_expander,
            local_hjr_stepper=local_hjr_stepper,
            active_set_post_filter=active_set_post_filter,
            break_criteria_checker=break_criteria_checker,
            verbose=verbose,
        )
