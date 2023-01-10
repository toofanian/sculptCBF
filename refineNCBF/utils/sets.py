from typing import List, Tuple

import hj_reachability
import numpy as np
import skfmm
import jax.numpy as jnp
from tqdm import tqdm

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


def map_cells_to_grid(
        cell_centerpoints: List[Vector],
        cell_halfwidths: Tuple[float, ...],
        grid: hj_reachability.Grid,
        verbose: bool = False,
        save_array: bool = False
) -> MaskNd:
    bool_grid = jnp.zeros_like(grid.states[..., 0], dtype=bool)
    for call_centerpoint in tqdm(cell_centerpoints, disable=not verbose, desc='mapping uncertified cells to grid'):
        for dim in range(grid.states.shape[-1]):
            bool_grid = (
                    bool_grid
                    |
                    (
                        (grid.states[..., dim] >= call_centerpoint[dim] - cell_halfwidths[dim])
                        &
                        (grid.states[..., dim] <= call_centerpoint[dim] + cell_halfwidths[dim])
                    )
            )

    if save_array:
        np.save(construct_full_path(generate_unique_filename('data/trained_NCBFs/uncertified_grid', 'npy')), bool_grid)

    return bool_grid


# @functools.partial(jax.jit, static_argnames=("cell_centerpoints", "cell_halfwidths"))
# def map_cells_to_grid_parallel_jax(
#         cell_centerpoints: Tuple[Vector],
#         cell_halfwidths: Tuple[float, ...],
#         grid: hj_reachability.Grid,
# ) -> MaskNd:
#     def update_grid_if_in_cell(_ind: int, _bool_grid: MaskNd):
#         def update_grid_at_dim(dim: int, __bool_grid: MaskNd):
#             grid_states_at_dim = grid.states[..., dim]
#             cell_centerpoints_at_index_and_dim = cell_centerpoints[_ind][dim]
#             cell_halfwidths_at_dim = cell_halfwidths[dim]
#
#             return (
#                 __bool_grid
#                 |
#                 (
#                     (grid_states_at_dim >= cell_centerpoints_at_index_and_dim - cell_halfwidths_at_dim)
#                     &
#                     grid_states_at_dim <= cell_centerpoints_at_index_and_dim + cell_halfwidths_at_dim
#                 )
#             )
#
#             # return (
#             #         __bool_grid
#             #         |
#             #         (
#             #             (grid.states[..., dim] >= cell_centerpoints[_ind][dim] - cell_halfwidths[dim])
#             #             &
#             #             (grid.states[..., dim] <= cell_centerpoints[_ind][dim] + cell_halfwidths[dim])
#             #         )
#             # )
#
#         _bool_grid = jax.lax.fori_loop(
#             lower=0, upper=grid.states.shape[-1],
#             body_fun=update_grid_at_dim,
#             init_val=_bool_grid
#         )
#         return _bool_grid
#
#     bool_grid = jnp.zeros_like(grid.states[..., 0], dtype=bool)
#     bool_grid = jax.lax.fori_loop(
#         lower=0, upper=len(cell_centerpoints),
#         body_fun=update_grid_if_in_cell,
#         init_val=bool_grid
#     )
#     return bool_grid
