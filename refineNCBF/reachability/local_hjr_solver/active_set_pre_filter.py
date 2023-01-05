from abc import ABC, abstractmethod
from typing import Callable

import attr

from verify_ncbf.toof.src.reachability.reachability_utils.results import LocalUpdateResult
from verify_ncbf.toof.src.utils.carving.masks import shrink_mask_by_signed_distance
from verify_ncbf.toof.src.utils.typing.types import MaskNd


@attr.s(auto_attribs=True)
class ActiveSetPreFilter(ABC, Callable):
    @abstractmethod
    def __call__(self, data: LocalUpdateResult) -> MaskNd:
        ...


@attr.s(auto_attribs=True)
class NoFilter(ActiveSetPreFilter):
    @classmethod
    def from_parts(cls):
        return cls()

    def __call__(self, data: LocalUpdateResult) -> MaskNd:
        active_set_filtered = data.get_recent_active_set()
        return active_set_filtered


@attr.s(auto_attribs=True)
class FilterWhereFarFromZeroLevelset(ActiveSetPreFilter):
    _distance: float

    @classmethod
    def from_parts(cls, distance: float):
        return cls(distance=distance)

    def __call__(self, data: LocalUpdateResult) -> MaskNd:
        active_set_filtered = (
                data.get_recent_active_set()
                & ~
                (
                        shrink_mask_by_signed_distance(data.get_recent_values() >= 0, distance=self._distance)
                        |
                        shrink_mask_by_signed_distance(data.get_recent_values() < 0, distance=self._distance)
                )
        )
        return active_set_filtered
