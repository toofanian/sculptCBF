from abc import ABC, abstractmethod
from typing import List, Callable

import attr
import jax.numpy as jnp

from refineNCBF.refining.local_hjr_solver.local_hjr_result import LocalUpdateResult


@attr.s(auto_attribs=True)
class BreakCriterion(ABC, Callable):
    @abstractmethod
    def __call__(self, data: LocalUpdateResult) -> bool:
        ...


@attr.s(auto_attribs=True)
class BreakCriteriaChecker(ABC, Callable):
    _break_criteria: List[BreakCriterion]

    @classmethod
    def from_criteria(cls, break_criteria: List[BreakCriterion]):
        return cls(break_criteria=break_criteria)

    def __call__(self, data: LocalUpdateResult) -> bool:
        check_results = (criterion(data) for criterion in self._break_criteria)
        if any(check_results):
            return True


@attr.s(auto_attribs=True)
class MaxIterations(BreakCriterion):
    _max_iterations: int

    @classmethod
    def from_parts(cls, max_iterations: int):
        return cls(max_iterations=max_iterations)

    def __call__(self, data: LocalUpdateResult) -> bool:
        return len(data) >= self._max_iterations


@attr.s(auto_attribs=True)
class PostFilteredActiveSetEmpty(BreakCriterion):
    @classmethod
    def from_parts(cls):
        return cls()

    def __call__(self, data: LocalUpdateResult) -> bool:
        return jnp.count_nonzero(data.get_recent_active_set()) == 0
