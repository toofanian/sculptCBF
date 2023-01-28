from abc import ABC, abstractmethod
from typing import Callable

import attr

from refineNCBF.refining.local_hjr_solver.local_hjr_result import LocalUpdateResult
from refineNCBF.utils.sets import shrink_mask_by_signed_distance
from refineNCBF.utils.types import MaskNd


@attr.s(auto_attribs=True)
class ActiveSetPreFilter(ABC, Callable):
    @abstractmethod
    def __call__(self, data: LocalUpdateResult) -> MaskNd:
        ...


@attr.s(auto_attribs=True)
class NoPreFilter(ActiveSetPreFilter):
    @classmethod
    def from_parts(cls):
        return cls()

    def __call__(self, data: LocalUpdateResult) -> MaskNd:
        active_set_filtered = data.get_recent_set_input()
        return active_set_filtered


@attr.s(auto_attribs=True)
class PreFilterWhereFarFromZeroLevelset(ActiveSetPreFilter):
    _distance: float

    @classmethod
    def from_parts(cls, distance: float):
        return cls(distance=distance)

    def __call__(self, data: LocalUpdateResult) -> MaskNd:
        recent_set_input = data.get_recent_set_input()
        where_far_exterior = shrink_mask_by_signed_distance(data.get_recent_values() >= 0, distance=self._distance)
        where_far_interior = shrink_mask_by_signed_distance(data.get_recent_values() < 0, distance=self._distance)

        active_set_filtered = (
                data.get_recent_set_input()
                & ~
                (
                        where_far_exterior
                        |
                        where_far_interior
                )
        )
        return active_set_filtered
