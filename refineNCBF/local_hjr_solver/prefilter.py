from abc import ABC, abstractmethod
from typing import Callable

import attr

from refineNCBF.local_hjr_solver.result import LocalUpdateResult
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
        active_set_filtered = data.get_pending_seed_set()
        return active_set_filtered


@attr.s(auto_attribs=True)
class PreFilterWhereFarFromZeroLevelset(ActiveSetPreFilter):
    _distance: float

    @classmethod
    def from_parts(cls, distance: float):
        return cls(distance=distance)

    def __call__(self, data: LocalUpdateResult) -> MaskNd:
        where_far_exterior = shrink_mask_by_signed_distance(data.get_recent_values() >= 0, distance=self._distance)
        where_far_interior = shrink_mask_by_signed_distance(data.get_recent_values() < 0, distance=self._distance)

        active_set_filtered = (
                data.get_pending_seed_set()
                & ~
                (
                        where_far_exterior
                        |
                        where_far_interior
                )
        )
        return active_set_filtered


@attr.s(auto_attribs=True)
class PreFilterWhereOutsideZeroLevelset(ActiveSetPreFilter):

    @classmethod
    def from_parts(cls):
        return cls()

    def __call__(self, data: LocalUpdateResult) -> MaskNd:
        active_set_filtered = (
                data.get_pending_seed_set()
                &
                data.get_viability_kernel()
        )
        return active_set_filtered


@attr.s(auto_attribs=True)
class PreFilterWhereFarFromBoundarySplit(ActiveSetPreFilter):
    _distance_inner: float
    _distance_outer: float

    @classmethod
    def from_parts(cls, distance_inner: float, distance_outer: float):
        return cls(distance_inner=distance_inner, distance_outer=distance_outer)

    def __call__(self, data: LocalUpdateResult) -> MaskNd:
        where_far_exterior = shrink_mask_by_signed_distance(data.get_recent_values() >= 0,
                                                            distance=self._distance_inner)
        where_far_interior = shrink_mask_by_signed_distance(data.get_recent_values() < 0, distance=self._distance_outer)

        active_set_filtered = (
                data.get_pending_seed_set()
                & ~
                (
                        where_far_exterior
                        |
                        where_far_interior
                )
        )
        return active_set_filtered


@attr.s(auto_attribs=True)
class PreFilterWhereFarFromBoundarySplitOnce(ActiveSetPreFilter):
    _distance_inner: float
    _distance_outer: float

    @classmethod
    def from_parts(cls, distance_inner: float, distance_outer: float):
        return cls(distance_inner=distance_inner, distance_outer=distance_outer)

    def __call__(self, data: LocalUpdateResult) -> MaskNd:
        if len(data) == 0:
            where_far_exterior = shrink_mask_by_signed_distance(data.get_recent_values() >= 0,
                                                                distance=self._distance_inner)
            where_far_interior = shrink_mask_by_signed_distance(data.get_recent_values() < 0,
                                                                distance=self._distance_outer)

            active_set_filtered = (
                    data.get_pending_seed_set()
                    & ~
                    (
                            where_far_exterior
                            |
                            where_far_interior
                    )
            )
            return active_set_filtered
        else:
            return data.get_pending_seed_set()
