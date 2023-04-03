import logging
from typing import Tuple

import attr
import numpy as np

from refineNCBF.utils.types import ArrayNd


@attr.dataclass
class DimName:
    dim: int
    name: str


@attr.dataclass
class ArraySlice1D:
    slice_index: Tuple
    slice_string: str
    free_dim_1: DimName

    @classmethod
    def from_reference_index(cls, reference_index: Tuple[int, ...], free_dim_1: DimName):
        slice_indices = []
        slice_string = "("
        for dim, slice_index in enumerate(reference_index):
            if dim == free_dim_1.dim:
                slice_indices.append(np.s_[:])
                slice_string = slice_string + ":, "
            else:
                slice_indices.append(int(slice_index))
                slice_string = slice_string + f"{int(slice_index)}, "
        slice_string = slice_string.rstrip() + ")"
        return cls(slice_index=tuple(slice_indices), slice_string=slice_string, free_dim_1=free_dim_1)

    def __len__(self):
        return len(self.slice_index)

    def get_sliced_array(self, array: ArrayNd) -> ArrayNd:
        return np.array(array)[self.slice_index]


@attr.dataclass
class ArraySlice2D:
    slice_index: Tuple
    slice_string: str
    free_dim_1: DimName
    free_dim_2: DimName

    @classmethod
    def from_array_slice_1d(cls, array_slice_1d: ArraySlice1D):
        return cls(
            slice_index=array_slice_1d.slice_index,
            slice_string=array_slice_1d.slice_string,
            free_dim_1=array_slice_1d.free_dim_1,
            free_dim_2=array_slice_1d.free_dim_1,
        )

    @classmethod
    def from_reference_index(cls, reference_index: Tuple[int, ...], free_dim_1: DimName, free_dim_2: DimName):
        slice_indices = []
        slice_string = "("
        for dim, slice_index in enumerate(reference_index):
            if dim == free_dim_1.dim or dim == free_dim_2.dim:
                slice_indices.append(np.s_[:])
                slice_string = slice_string + ":, "
            else:
                slice_indices.append(int(slice_index))
                slice_string = slice_string + f"{int(slice_index)}, "
        slice_string = slice_string.rstrip() + ")"
        return cls(
            slice_index=tuple(slice_indices),
            slice_string=slice_string,
            free_dim_1=free_dim_1,
            free_dim_2=free_dim_2,
        )

    def __len__(self):
        return len(self.slice_index)

    def get_sliced_array(self, array: ArrayNd) -> ArrayNd:
        vals = self._protect_2d_if_same_dimensions(np.array(array)[self.slice_index])
        if self.free_dim_1.dim > self.free_dim_2.dim:
            vals = vals.T
        return vals

    def _protect_2d_if_same_dimensions(self, values: ArrayNd) -> ArrayNd:
        if self.free_dim_1.dim == self.free_dim_2.dim:
            values = np.tile(values, (values.size, 1))
        return values


def make_configured_logger(name: str) -> logging.Logger:
    logging.basicConfig()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
