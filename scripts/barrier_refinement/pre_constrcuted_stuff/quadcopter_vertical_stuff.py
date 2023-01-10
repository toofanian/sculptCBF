from typing import Callable

import hj_reachability
import numpy as np
import torch

from refineNCBF.dynamic_systems.implementations.quadcopter import quadcopter_vertical_jax_hj
from refineNCBF.refining.hj_reachability_interface.hj_setup import HjSetup

import jax.numpy as jnp

from refineNCBF.training.dnn_models.standardizer import Standardizer
from refineNCBF.utils.types import VectorBatch, ScalarBatch, ArrayNd


def make_hj_setup_quadcopter_vertical() -> HjSetup:
    dynamics = quadcopter_vertical_jax_hj
    grid = hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj_reachability.sets.Box(
            [0, -8, -jnp.pi, -10],
            [10, 8, jnp.pi, 10]
        ),
        shape=(21, 21, 21, 21)
    )
    return HjSetup.from_parts(dynamics=dynamics, grid=grid)


def quadcopter_cbf_from_refine_cbf(state: VectorBatch) -> ScalarBatch:
    scaling = jnp.array([0.75, 0.5, 2., 0.5])
    return 15 - (scaling[0] * (5 - state[..., 0]) ** 2 + scaling[1] * (state[..., 1]) ** 2
                 + scaling[2] * (state[..., 2]) ** 2 + scaling[3] * (state[..., 3]) ** 2)


def tabularize_vector_to_scalar_mapping(
        mapping: Callable[[VectorBatch], ScalarBatch],
        grid: hj_reachability.Grid
) -> ArrayNd:
    return mapping(grid.states.reshape((-1, grid.states.shape[-1]))).reshape(grid.shape)


def tabularize_dnn(
        dnn: Callable[[VectorBatch], ScalarBatch],
        standardizer: Standardizer,
        grid: hj_reachability.Grid
) -> ArrayNd:
    return jnp.array(dnn((torch.FloatTensor(standardizer.standardize(np.array(grid.states.reshape((-1, grid.states.shape[-1]))))))).detach().numpy().reshape(grid.shape))
