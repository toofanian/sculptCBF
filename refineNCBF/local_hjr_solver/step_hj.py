from abc import ABC, abstractmethod
from typing import Callable

import attr

import hj_reachability
from hj_reachability.solver import backwards_reachable_tube
from refineNCBF.hj_reachability_interface.hj_value_postprocessors import ReachAvoid
from refineNCBF.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.utils.sets import compute_signed_distance
from refineNCBF.utils.types import MaskNd, ArrayNd


@attr.s(auto_attribs=True)
class LocalHjrStepper(ABC, Callable):
    @abstractmethod
    def __call__(
            self,
            data: LocalUpdateResult,
            active_set_prefiltered: MaskNd,
            active_set_expanded: MaskNd
    ) -> ArrayNd:
        ...


@attr.s(auto_attribs=True)
class ClassicLocalHjrStepper(LocalHjrStepper):
    _dynamics: hj_reachability.Dynamics
    _grid: hj_reachability.Grid
    _solver_settings: hj_reachability.SolverSettings
    _time_step: float
    _verbose: bool

    @classmethod
    def from_parts(
            cls,
            dynamics: hj_reachability.Dynamics,
            grid: hj_reachability.Grid,
            terminal_values: ArrayNd,
            time_step: float,
            verbose: bool, 
            accuracy: hj_reachability.solver.SolverAccuracyEnum = hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH
    ):
        solver_settings = hj_reachability.SolverSettings.with_accuracy(
            accuracy,
            value_postprocessor=ReachAvoid.from_array(
                values=terminal_values,
            ),
        )
        return cls(
            dynamics=dynamics,
            grid=grid,
            solver_settings=solver_settings,
            time_step=time_step,
            verbose=verbose
        )

    def __call__(self, data: LocalUpdateResult, active_set_prefiltered: MaskNd, active_set_expanded: MaskNd) -> ArrayNd:
        values = hj_reachability.step(
            solver_settings=self._solver_settings,
            dynamics=self._dynamics,
            grid=self._grid,
            time=0,
            values=data.get_recent_values(),
            target_time=self._time_step,
            active_set=active_set_expanded,
            progress_bar=self._verbose,
        )
        return values


@attr.s(auto_attribs=True)
class DecreaseLocalHjrStepper(LocalHjrStepper):
    _dynamics: hj_reachability.Dynamics
    _grid: hj_reachability.Grid
    _solver_settings: hj_reachability.SolverSettings
    _time_step: float
    _verbose: bool

    @classmethod
    def from_parts(
            cls,
            dynamics: hj_reachability.Dynamics,
            grid: hj_reachability.Grid,
            terminal_values: ArrayNd,
            time_step: float,
            verbose: bool,
            accuracy: hj_reachability.solver.SolverAccuracyEnum = hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH
    ):
        solver_settings = hj_reachability.SolverSettings.with_accuracy(
            accuracy,
            value_postprocessor=ReachAvoid.from_array(
                values=terminal_values,
            ),
            hamiltonian_postprocessor=backwards_reachable_tube
        )
        return cls(
            dynamics=dynamics,
            grid=grid,
            solver_settings=solver_settings,
            time_step=time_step,
            verbose=verbose
        )

    def __call__(
            self,
            data: LocalUpdateResult,
            active_set_prefiltered: MaskNd,
            active_set_expanded: MaskNd
    ) -> ArrayNd:
        values_next = hj_reachability.step(
            solver_settings=self._solver_settings,
            dynamics=self._dynamics,
            grid=self._grid,
            time=0,
            values=data.get_recent_values(),
            target_time=self._time_step,
            active_set=active_set_expanded,
            progress_bar=self._verbose,
        )
        return values_next


@attr.s(auto_attribs=True)
class DecreaseReplaceLocalHjrStepper(LocalHjrStepper):
    _dynamics: hj_reachability.Dynamics
    _grid: hj_reachability.Grid
    _solver_settings: hj_reachability.SolverSettings
    _time_step: float
    _verbose: bool

    @classmethod
    def from_parts(
            cls,
            dynamics: hj_reachability.Dynamics,
            grid: hj_reachability.Grid,
            terminal_values: ArrayNd,
            time_step: float,
            verbose: bool,
            accuracy: hj_reachability.solver.SolverAccuracyEnum = hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH
    ):
        solver_settings = hj_reachability.SolverSettings.with_accuracy(
            accuracy,
            value_postprocessor=ReachAvoid.from_array(
                values=terminal_values,
            ),
            hamiltonian_postprocessor=backwards_reachable_tube
        )
        return cls(dynamics=dynamics, grid=grid, solver_settings=solver_settings, time_step=time_step, verbose=verbose)

    def __call__(self, data: LocalUpdateResult, active_set_prefiltered: MaskNd, active_set_expanded: MaskNd) -> ArrayNd:
        values_next = hj_reachability.step(
            solver_settings=self._solver_settings,
            dynamics=self._dynamics,
            grid=self._grid,
            time=0,
            values=data.get_recent_values(),
            target_time=self._time_step,
            active_set=active_set_expanded,
            progress_bar=self._verbose,
        )
        values_decreased = (values_next < data.get_recent_values())  # & active_set_expanded
        values = data.get_recent_values().at[values_decreased].set(values_next[values_decreased])
        signed_distance_to_kernel = compute_signed_distance(values >= 0)
        values = values.at[signed_distance_to_kernel >= 3].set(
            signed_distance_to_kernel[signed_distance_to_kernel >= 3])
        values = values.at[signed_distance_to_kernel < -3].set(
            signed_distance_to_kernel[signed_distance_to_kernel < -3])
        return values


@attr.s(auto_attribs=True)
class TrashLocalHjrStepper(LocalHjrStepper):
    """
    used when testing hjr solver performance. computes update but returns input values, so nothing changes in the pipeline,
    and it can be repeated identically for another iteration
    """
    _dynamics: hj_reachability.Dynamics
    _grid: hj_reachability.Grid
    _solver_settings: hj_reachability.SolverSettings
    _time_step: float
    _verbose: bool

    @classmethod
    def from_parts(
            cls,
            dynamics: hj_reachability.Dynamics,
            grid: hj_reachability.Grid,
            solver_settings: hj_reachability.SolverSettings,
            time_step: float,
            verbose: bool
    ):
        return cls(dynamics=dynamics, grid=grid, solver_settings=solver_settings, time_step=time_step, verbose=verbose)

    def __call__(self, data: LocalUpdateResult, active_set_prefiltered: MaskNd, active_set_expanded: MaskNd) -> ArrayNd:
        values_next = hj_reachability.step(
            solver_settings=self._solver_settings,
            dynamics=self._dynamics,
            grid=self._grid,
            time=0,
            values=data.get_recent_values(),
            target_time=self._time_step,
            active_set=active_set_expanded,
            progress_bar=self._verbose
        )
        return data.get_recent_values()
