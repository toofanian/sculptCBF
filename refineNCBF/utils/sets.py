from typing import List, Tuple

import hj_reachability
import numpy as np
import skfmm
import jax.numpy as jnp
from tqdm import tqdm
import sys

from refineNCBF.utils.files import generate_unique_filename, construct_full_path
from refineNCBF.utils.types import MaskNd, ArrayNd, Vector


# TODO default distance should be .5, since it is from boundary of cell to center of cell


def compute_signed_distance(bool_array: MaskNd) -> ArrayNd:
    signed_distance = jnp.array(-skfmm.distance(~bool_array) + skfmm.distance(bool_array))
    return signed_distance


def expand_mask_by_signed_distance(mask: MaskNd, distance: float = 1) -> MaskNd:
    if np.count_nonzero(mask) == 0 or np.count_nonzero(~mask) == 0:
        print('mask is full, returning mask')
        return mask
    signed_distance = compute_signed_distance(mask)
    expanded_mask = (signed_distance >= -distance)
    return expanded_mask


def shrink_mask_by_signed_distance(mask: MaskNd, distance: float = 1) -> MaskNd:
    if np.count_nonzero(mask) == 0 or np.count_nonzero(~mask) == 0:
        print('mask is full, returning mask')
        return mask
    signed_distance = compute_signed_distance(mask)
    shrunk_mask = (signed_distance >= distance)
    return shrunk_mask


def get_mask_boundary_by_signed_distance(mask: MaskNd, distance: float = 1) -> MaskNd:
    if np.count_nonzero(mask) == 0 or np.count_nonzero(~mask) == 0:
        print('mask is full, returning mask')
        return mask
    signed_distance = compute_signed_distance(mask)
    mask_boundary = (signed_distance <= distance) & (signed_distance >= 0)
    return mask_boundary


def get_mask_boundary_on_both_sides_by_signed_distance(mask: MaskNd, distance: float = 1) -> MaskNd:
    if np.count_nonzero(mask) == 0 or np.count_nonzero(~mask) == 0:
        print('mask is full, returning mask')
        return mask
    signed_distance = compute_signed_distance(mask)
    mask_boundary = (signed_distance <= distance) & (signed_distance >= -distance)
    return mask_boundary


def map_cells_to_grid(
        cell_centerpoints: List[Vector],
        cell_halfwidths: Tuple[float, ...],
        grid: hj_reachability.Grid,
        verbose: bool = False,
        save_array: bool = False
) -> MaskNd:
    # TODO: parallelize this!
    bool_grid = jnp.zeros_like(grid.states[..., 0], dtype=bool)
    for call_centerpoint in tqdm(cell_centerpoints, disable=not verbose, desc='mapping uncertified cells to grid'):
        grid_within_cell = jnp.ones_like(bool_grid, dtype=bool)
        for dim in range(grid.states.shape[-1]):
            grid_where_greater = (grid.states[..., dim] + grid.spacings[dim]) >= (call_centerpoint[dim] - cell_halfwidths[dim])
            grid_where_lesser = (grid.states[..., dim] - grid.spacings[dim]) <= (call_centerpoint[dim] + cell_halfwidths[dim])
            grid_within_cell = grid_within_cell & (grid_where_greater & grid_where_lesser)
        bool_grid = bool_grid | grid_within_cell

    if save_array:
        np.save(construct_full_path(generate_unique_filename('data/trained_NCBFs/quad4d_boundary/uncertified_grid', 'npy')), bool_grid)

    return bool_grid
