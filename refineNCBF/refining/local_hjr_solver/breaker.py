import logging
from abc import ABC, abstractmethod
from typing import List, Callable

import attr
import jax.numpy as jnp

from refineNCBF.refining.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.utils.visuals import make_configured_logger


@attr.s(auto_attribs=True)
class BreakCriterion(ABC, Callable):
    @abstractmethod
    def __call__(self, data: LocalUpdateResult) -> bool:
        ...

    @abstractmethod
    def get_descriptor(self) -> str:
        ...


@attr.s(auto_attribs=True)
class BreakCriteriaChecker(ABC, Callable):
    _break_criteria: List[BreakCriterion]

    _verbose: bool
    _logger: logging.Logger = make_configured_logger(__name__)

    @classmethod
    def from_criteria(cls, break_criteria: List[BreakCriterion], verbose: bool = False):
        return cls(break_criteria=break_criteria, verbose=verbose)

    def __call__(self, data: LocalUpdateResult) -> bool:
        break_reasons = self._get_break_reasons(data)
        criterion_met = len(break_reasons) > 0

        if criterion_met:
            if self._verbose:
                self._logger.info(f"Breaking because of: {break_reasons}")

        return criterion_met

    def _get_break_reasons(self, data):
        break_reasons = [criterion.get_descriptor() for criterion in self._break_criteria if criterion(data)]
        return break_reasons


@attr.s(auto_attribs=True)
class MaxIterations(BreakCriterion):
    _max_iterations: int

    @classmethod
    def from_parts(cls, max_iterations: int):
        return cls(max_iterations=max_iterations)

    def __call__(self, data: LocalUpdateResult) -> bool:
        return len(data) >= self._max_iterations

    def get_descriptor(self) -> str:
        return f'criterion of maximum {self._max_iterations} iterations has been met'


@attr.s(auto_attribs=True)
class PostFilteredActiveSetEmpty(BreakCriterion):
    @classmethod
    def from_parts(cls):
        return cls()

    def __call__(self, data: LocalUpdateResult) -> bool:
        return jnp.count_nonzero(data.get_pending_seed_set()) == 0

    def get_descriptor(self) -> str:
        return f'criterion of empty post-filtered active set has been met'