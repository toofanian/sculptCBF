from abc import ABC, abstractmethod
from typing import Callable

import attr
import jax

from refineNCBF.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.utils.sets import expand_mask_by_signed_distance, compute_signed_distance, expand_mask_by_dilation, \
    get_mask_boundary_by_dilation
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
        if self._distance == jax.numpy.inf:
            return jax.numpy.ones_like(source_set, dtype=bool)

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
            signed_distance_active = compute_signed_distance(source_set)
            signed_distance_kernel = compute_signed_distance(data.get_viability_kernel())
            expanded = signed_distance_active >= -self._neighbor_distance
            boundary = (signed_distance_kernel <= self._boundary_distance_inner) & (
                        signed_distance_kernel >= -self._boundary_distance_outer)
            active_set_expanded = expanded & boundary
            # expanded = expand_mask_by_signed_distance(
            #     source_set,
            #     self._neighbor_distance
            # )
            # boundary_inner = get_mask_boundary_by_signed_distance(
            #     data.get_viability_kernel(),
            #     self._boundary_distance_inner
            # )
            # boundary_outer = get_mask_boundary_by_signed_distance(
            #     ~data.get_viability_kernel(),
            #     self._boundary_distance_outer
            # )
            # active_set_expanded = expanded & (boundary_inner | boundary_outer)
        return active_set_expanded


@attr.s(auto_attribs=True)
class SignedDistanceNeighborsNearBoundaryDilation(NeighborExpander):
    _neighbor_distance: int
    _boundary_distance_inner: int
    _boundary_distance_outer: int

    @classmethod
    def from_parts(
            cls,
            neighbor_distance: int,
            boundary_distance_inner: int,
            boundary_distance_outer: int
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
            expanded = expand_mask_by_dilation(source_set, self._neighbor_distance)
            boundary = get_mask_boundary_by_dilation(data.get_viability_kernel(), self._boundary_distance_inner,
                                                     self._boundary_distance_outer)
            active_set_expanded = expanded & boundary
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
