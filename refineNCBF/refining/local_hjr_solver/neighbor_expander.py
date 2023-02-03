from abc import ABC, abstractmethod
from typing import Callable

import attr

from refineNCBF.refining.local_hjr_solver.local_hjr_result import LocalUpdateResult
from refineNCBF.utils.sets import expand_mask_by_signed_distance
from refineNCBF.utils.types import MaskNd


@attr.s(auto_attribs=True)
class NeighborExpander(ABC, Callable):
    @abstractmethod
    def __call__(self, data: LocalUpdateResult, source_set: MaskNd) -> MaskNd:
        ...


@attr.s(auto_attribs=True)
class SignedDistanceNeighbors(NeighborExpander):
    _distance: float

    @classmethod
    def from_parts(cls, distance: float):
        return cls(distance=distance)

    def __call__(self, data: LocalUpdateResult, source_set: MaskNd) -> MaskNd:
        if len(data) == 0:
            return source_set
        else:
            return expand_mask_by_signed_distance(source_set, self._distance)



@attr.s(auto_attribs=True)
class InnerSignedDistanceNeighbors(NeighborExpander):
    _distance: float

    @classmethod
    def from_parts(cls, distance: float):
        return cls(distance=distance)

    def __call__(self, data: LocalUpdateResult, source_set: MaskNd) -> MaskNd:
        inner_set = data.get_viability_kernel()
        expanded_set = expand_mask_by_signed_distance(source_set, self._distance) & inner_set
        return expanded_set


@attr.s(auto_attribs=True)
class NoNeighbors(NeighborExpander):

    @classmethod
    def from_parts(cls):
        return cls()

    def __call__(self, data: LocalUpdateResult, source_set: MaskNd) -> MaskNd:
        return source_set


# TODO: add a neighbor expander that uses dynamics.
#  use partial max magnitudes to determine the maximum step size from this cell in each direction.
#  then snap to nearest grid point in each direction using snap_state_to_grid_index(), and fill in the box between the 2^N points.

