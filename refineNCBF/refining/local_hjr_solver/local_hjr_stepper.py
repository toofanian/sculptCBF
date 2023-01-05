from abc import ABC, abstractmethod
from typing import Callable

import attr
import hj_reachability

from verify_ncbf.toof.src.reachability.reachability_utils.hj_setup import HjSetup
from verify_ncbf.toof.src.reachability.reachability_utils.local_utils import hjr_step_local
from verify_ncbf.toof.src.reachability.reachability_utils.results import LocalUpdateResult
from verify_ncbf.toof.src.utils.typing.types import ArrayNd, MaskNd


@attr.s(auto_attribs=True)
class LocalHjrStepper(ABC, Callable):
    @abstractmethod
    def __call__(self, data: LocalUpdateResult, active_set_prefiltered: MaskNd, active_set_expanded: MaskNd) -> ArrayNd:
        ...


@attr.s(auto_attribs=True)
class ClassicLocalHjrStepper(LocalHjrStepper):
    _hj_setup: HjSetup
    _solver_settings: hj_reachability.SolverSettings
    _time_step: float
    _verbose: bool

    @classmethod
    def from_parts(cls, hj_setup: HjSetup, solver_settings: hj_reachability.SolverSettings, time_step: float,
                   verbose: bool):
        return cls(hj_setup=hj_setup, solver_settings=solver_settings, time_step=time_step, verbose=verbose)

    def __call__(self, data: LocalUpdateResult, active_set_prefiltered: MaskNd, active_set_expanded: MaskNd) -> ArrayNd:
        values = hjr_step_local(
            solver_settings=self._solver_settings,
            hj_setup=self._hj_setup,
            start_time=0,
            values=data.get_recent_values(),
            target_time=self._time_step,
            active_set=active_set_expanded,
            progress_bar=self._verbose
        )
        return values


@attr.s(auto_attribs=True)
class OnlyDecreaseLocalHjrStepper(LocalHjrStepper):
    _hj_setup: HjSetup
    _solver_settings: hj_reachability.SolverSettings
    _time_step: float
    _verbose: bool

    @classmethod
    def from_parts(cls, hj_setup: HjSetup, solver_settings: hj_reachability.SolverSettings, time_step: float,
                   verbose: bool):
        return cls(hj_setup=hj_setup, solver_settings=solver_settings, time_step=time_step, verbose=verbose)

    def __call__(self, data: LocalUpdateResult, active_set_prefiltered: MaskNd, active_set_expanded: MaskNd) -> ArrayNd:
        values_next = hjr_step_local(
            solver_settings=self._solver_settings,
            hj_setup=self._hj_setup,
            start_time=0,
            values=data.get_recent_values(),
            target_time=self._time_step,
            active_set=active_set_expanded,
            progress_bar=self._verbose
        )
        values_decreased = values_next < data.get_recent_values()
        values = data.get_recent_values().at[values_decreased].set(values_next[values_decreased])
        return values
