import logging
from typing import Tuple

import attr
import numpy as np

from refineNCBF.utils.files import generate_unique_filename
from refineNCBF.utils.types import ArrayNd


@attr.dataclass
class ArraySlice2D:
    slice_index: Tuple
    free_dim_1: int
    free_dim_2: int

    @classmethod
    def from_reference_index(cls, reference_index: Tuple[int, ...], free_dim_1: int, free_dim_2: int):
        slice_indices = []
        for dim, slice_index in enumerate(reference_index):
            if dim == free_dim_1 or dim == free_dim_2:
                slice_indices.append(np.s_[:])
            else:
                slice_indices.append(slice_index)
        return cls(slice_index=tuple(slice_indices), free_dim_1=free_dim_1, free_dim_2=free_dim_2)

    def __len__(self):
        return len(self.slice_index)

    def get_sliced_array(self, array: ArrayNd) -> ArrayNd:
        return np.array(array)[self.slice_index]


def make_configured_logger(name: str) -> logging.Logger:
    logging.basicConfig()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
