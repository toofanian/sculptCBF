from typing import Callable, Optional, Tuple

import hj_reachability
import numpy as np
import torch
from jax import numpy as jnp
from tqdm import tqdm

from refineNCBF.training.dnn_models.standardizer import Standardizer
from refineNCBF.utils.files import construct_full_path, generate_unique_filename
from refineNCBF.utils.types import VectorBatch, ScalarBatch, ArrayNd, MaskNd, Vector


def tabularize_vector_to_scalar_mapping(
        mapping: Callable[[VectorBatch], ScalarBatch],
        grid: hj_reachability.Grid
) -> ArrayNd:
    return mapping(grid.states.reshape((-1, grid.states.shape[-1]))).reshape(grid.shape)


def tabularize_dnn(
        dnn: Callable[[VectorBatch], ScalarBatch],
        grid: hj_reachability.Grid,
        standardizer: Optional[Standardizer] = None,
) -> ArrayNd:
    flat_states = np.array(grid.states.reshape((-1, grid.states.shape[-1])))
    if isinstance(standardizer, Standardizer):
        flat_states = standardizer.standardize(flat_states)
    tensor_states = torch.FloatTensor(flat_states)
    dnn_output = dnn(tensor_states)
    if isinstance(dnn_output, torch.Tensor):
        dnn_output = dnn_output.detach().numpy()
    output = jnp.array(dnn_output.reshape((*grid.shape, dnn_output.shape[-1]))).squeeze()
    return output


def flag_states_on_grid(
        cell_centerpoints: VectorBatch,
        cell_halfwidths: Tuple[float, ...],
        grid: hj_reachability.Grid,
        verbose: bool = False,
        save_array: bool = False
) -> MaskNd:
    dims = grid.states.shape[-1]

    cell_lower_bounds = cell_centerpoints - cell_halfwidths
    cell_upper_bounds = cell_centerpoints + cell_halfwidths

    cell_lower_bounds_in_grid_frame = cell_lower_bounds + np.array(grid.spacings).reshape((1, dims))/2 - np.array(grid.domain.lo).reshape((1, dims))
    cell_upper_bounds_in_grid_frame = cell_upper_bounds + np.array(grid.spacings).reshape((1, dims))/2 - np.array(grid.domain.lo).reshape((1, dims))

    lower_index = np.maximum(cell_lower_bounds_in_grid_frame // np.array(grid.spacings).reshape((1, dims)), 0)
    upper_index = np.minimum(cell_upper_bounds_in_grid_frame // np.array(grid.spacings).reshape((1, dims)), np.array(grid.states.shape[0:-1]).reshape((1, dims))-1)

    overlap_indices = np.stack([lower_index, upper_index], axis=-1).astype(int)

    bool_grid = np.zeros_like(grid.states[..., 0], dtype=bool)
    for overlap_index in tqdm(overlap_indices, disable=not verbose, desc='flagging states on grid'):
        index_slice = tuple([np.s_[overlap_index[dim][0]:overlap_index[dim][1]+1] for dim in range(dims)])
        bool_grid[index_slice] = True

    if save_array:
        np.save(construct_full_path(generate_unique_filename('data/trained_NCBFs/quad4d_boundary/uncertified_grid', 'npy')), bool_grid)

    return bool_grid


def snap_state_to_grid_index(
        state: Vector,
        grid: hj_reachability.Grid
) -> Tuple[int, ...]:
    state_in_grid_frame = state + np.array(grid.spacings)/2 - np.array(grid.domain.lo)
    grid_index = state_in_grid_frame // jnp.array(grid.spacings)
    tuple_output = tuple(grid_index.astype(int))
    return tuple_output
