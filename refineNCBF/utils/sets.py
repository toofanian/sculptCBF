from typing import List, Tuple

import hj_reachability
import numpy as np
import skfmm
import jax.numpy as jnp
from tqdm import tqdm
import sys

from refineNCBF.utils.files import generate_unique_filename, construct_full_path
from refineNCBF.utils.types import MaskNd, ArrayNd, Vector, VectorBatch


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
            grid_where_greater = (grid.states[..., dim] + grid.spacings[dim]*.5) >= (call_centerpoint[dim] - cell_halfwidths[dim])
            grid_where_lesser = (grid.states[..., dim] - grid.spacings[dim]*.5) <= (call_centerpoint[dim] + cell_halfwidths[dim])
            grid_within_cell = grid_within_cell & (grid_where_greater & grid_where_lesser)
        bool_grid = bool_grid | grid_within_cell

    if save_array:
        np.save(construct_full_path(generate_unique_filename('data/trained_NCBFs/quad4d_boundary/uncertified_grid', 'npy')), bool_grid)

    return bool_grid


def map_cells_to_grid_using_mod(
        cell_centerpoints: List[Vector],
        cell_halfwidths: Tuple[float, ...],
        grid: hj_reachability.Grid,
        verbose: bool = False,
        save_array: bool = False
) -> MaskNd:
    bool_grid = np.zeros_like(grid.states[..., 0], dtype=bool)
    for cell_centerpoint in tqdm(cell_centerpoints, disable=not verbose, desc='mapping uncertified cells to grid'):
        overlap_indices = []
        for dim in range(grid.states.shape[-1]):
            state_val_at_dim_min = cell_centerpoint[dim] - cell_halfwidths[dim]
            state_val_at_dim_max = cell_centerpoint[dim] + cell_halfwidths[dim]

            min_index = int(max((state_val_at_dim_min + grid.spacings[dim]*.5 - grid.domain.lo[dim]) // grid.spacings[dim], 0))
            max_index = int(min((state_val_at_dim_max + grid.spacings[dim]*.5 - grid.domain.lo[dim]) // grid.spacings[dim], grid.states.shape[dim] - 1))
            overlap_indices.append(np.s_[min_index:max_index+1])
        overlap_indices = tuple(overlap_indices)
        bool_grid[overlap_indices] = True


    if save_array:
        np.save(construct_full_path(generate_unique_filename('data/trained_NCBFs/quad4d_boundary/uncertified_grid', 'npy')), bool_grid)

    return bool_grid

def map_cells_to_grid_using_mod_parallel(
        cell_centerpoints: VectorBatch,
        cell_halfwidths: Tuple[float, ...],
        grid: hj_reachability.Grid,
        verbose: bool = False,
        save_array: bool = False
) -> MaskNd:
    dims = grid.states.shape[-1]

    cell_lower_bounds = cell_centerpoints - cell_halfwidths
    cell_upper_bounds = cell_centerpoints + cell_halfwidths

    cell_lower_bounds_in_grid_frame = cell_lower_bounds + np.array(grid.spacings).reshape((1, dims))*.5 - np.array(grid.domain.lo).reshape((1, dims))
    cell_upper_bounds_in_grid_frame = cell_upper_bounds + np.array(grid.spacings).reshape((1, dims))*.5 - np.array(grid.domain.lo).reshape((1, dims))

    lower_index = np.maximum(cell_lower_bounds_in_grid_frame // np.array(grid.spacings).reshape((1, dims)), 0)
    upper_index = np.minimum(cell_upper_bounds_in_grid_frame // np.array(grid.spacings).reshape((1, dims)), grid.states.shape[0] - 1)

    overlap_indices = np.stack([lower_index, upper_index], axis=-1).astype(int)

    bool_grid = np.zeros_like(grid.states[..., 0], dtype=bool)
    for overlap_index in tqdm(overlap_indices, disable=not verbose, desc='mapping uncertified cells to grid'):
        index_slice = tuple([np.s_[overlap_index[dim][0]:overlap_index[dim][1]+1] for dim in range(dims)])
        bool_grid[index_slice] = True

    if save_array:
        np.save(construct_full_path(generate_unique_filename('data/trained_NCBFs/quad4d_boundary/uncertified_grid', 'npy')), bool_grid)

    return bool_grid
