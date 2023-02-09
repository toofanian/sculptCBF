from abc import ABC, abstractmethod
from typing import Callable

import attr

from refineNCBF.refining.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.utils.sets import expand_mask_by_signed_distance, get_mask_boundary_on_both_sides_by_signed_distance, get_mask_boundary_by_signed_distance
from refineNCBF.utils.types import MaskNd


@attr.s(auto_attribs=True)
class NeighborExpander(ABC, Callable):
    @abstractmethod
    def __call__(
            self,
            data: LocalUpdateResult,
            source_set: MaskNd
    ) -> MaskNd:
        ...


@attr.s(auto_attribs=True)
class SignedDistanceNeighbors(NeighborExpander):
    _distance: float

    @classmethod
    def from_parts(
            cls,
            distance: float
    ):
        return cls(
            distance=distance
        )

    def __call__(
            self,
            data: LocalUpdateResult,
            source_set: MaskNd
    ) -> MaskNd:
        if len(data) == 0:
            active_set_expanded = source_set
        else:
            active_set_expanded = expand_mask_by_signed_distance(
                source_set,
                self._distance
            )
        return active_set_expanded


@attr.s(auto_attribs=True)
class SignedDistanceNeighborsNearBoundary(NeighborExpander):
    _neighbor_distance: float
    _boundary_distance_inner: float
    _boundary_distance_outer: float

    @classmethod
    def from_parts(
            cls,
            neighbor_distance: float,
            boundary_distance_inner: float,
            boundary_distance_outer: float
    ):
        return cls(
            neighbor_distance=neighbor_distance,
            boundary_distance_inner=boundary_distance_inner,
            boundary_distance_outer=boundary_distance_outer
        )

    def __call__(
            self,
            data: LocalUpdateResult,
            source_set: MaskNd
    ) -> MaskNd:
        if len(data) == 0:
            active_set_expanded = source_set
        else:
            expanded = expand_mask_by_signed_distance(
                source_set,
                self._neighbor_distance
            )
            boundary_inner = get_mask_boundary_by_signed_distance(
                data.get_viability_kernel(),
                self._boundary_distance_inner
            )
            boundary_outer = get_mask_boundary_by_signed_distance(
                ~data.get_viability_kernel(),
                self._boundary_distance_outer
            )
            active_set_expanded = expanded & (boundary_inner | boundary_outer)
        return active_set_expanded


@attr.s(auto_attribs=True)
class InnerSignedDistanceNeighbors(NeighborExpander):
    _distance: float

    @classmethod
    def from_parts(
            cls,
            distance: float
    ):
        return cls(
            distance=distance
        )

    def __call__(
            self,
            data: LocalUpdateResult,
            source_set: MaskNd
    ) -> MaskNd:
        if len(data) == 0:
            active_set_expanded = source_set
        else:
            active_set_expanded = expand_mask_by_signed_distance(
                source_set,
                self._distance
            ) & data.get_viability_kernel()
        return active_set_expanded


@attr.s(auto_attribs=True)
class NoNeighbors(NeighborExpander):
    @classmethod
    def from_parts(
            cls
    ):
        return cls()

    def __call__(
            self,
            data: LocalUpdateResult,
            source_set: MaskNd
    ) -> MaskNd:
        return source_set


# TODO: add a neighbor expander that uses dynamics.
#  use partial max magnitudes to determine the maximum step size from this cell in each direction.
#  then snap to nearest grid point in each direction using snap_state_to_grid_index(), and fill in the box between the 2^N points.

