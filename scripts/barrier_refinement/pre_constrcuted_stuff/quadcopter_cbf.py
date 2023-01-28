import json
from typing import Callable, Union

import attr
import hj_reachability
import numpy as np
import stable_baselines3
import torch

from neural_barrier_kinematic_model.custom_gym.envs.QuadVertical.quad_vertical import Quad_Vertical_Env as QuadVerticalEnv

from refineNCBF.training.dnn_models.cbf import Cbf
from refineNCBF.utils.files import construct_full_path
from refineNCBF.training.dnn_models.standardizer import Standardizer
from refineNCBF.utils.tables import tabularize_dnn, snap_state_to_grid_index
from refineNCBF.utils.types import VectorBatch, ArrayNd, Vector


def load_quadcopter_cbf() -> Cbf:
    device = 'cpu'
    print("loading CBF model...")
    cbf_func = Cbf(4, 256)
    cbf_ckpt = torch.load(construct_full_path('data/trained_NCBFs/quad4d_boundary/quad4d_cbf.pth'), map_location=device)
    cbf_func.load_state_dict(cbf_ckpt['model_state_dict'])
    cbf_func.to(device)
    return cbf_func


def load_standardizer() -> Standardizer:
    standardizer = Standardizer(fp=construct_full_path('data/trained_NCBFs/quad4d_boundary/quad_4d_standardizer.npy'))
    standardizer.initialize_from_file()
    return standardizer


def load_uncertified_states() -> VectorBatch:
    with open(construct_full_path('data/trained_NCBFs/quad4d_boundary/quad4d_boundary_cert_results.json')) as f:
        quad4d_result_dict = json.load(f)
    uncertified_states = quad4d_result_dict['uns']
    violated_states = quad4d_result_dict['vio']
    total_states = uncertified_states + violated_states

    standardizer = load_standardizer()
    total_states_destandardized = standardizer.destandardize(np.array(total_states))
    return total_states_destandardized


@attr.s(auto_attribs=True)
class StableBaselinesCallable(Callable):
    network: Union[stable_baselines3.PPO, stable_baselines3.SAC]

    def __call__(self, state):
        return self.network.predict(state, deterministic=True)[0]


def load_policy_ppo() -> StableBaselinesCallable:
    return StableBaselinesCallable(
        stable_baselines3.PPO.load(construct_full_path('data/trained_NCBFs/quad4d_boundary/best_model.zip'))
    )


def load_policy_sac() -> StableBaselinesCallable:
    return StableBaselinesCallable(
        stable_baselines3.SAC.load(construct_full_path('data/trained_NCBFs/quad4d_boundary/best_model-2.zip'),  env=QuadVerticalEnv())
    )


@attr.s(auto_attribs=True)
class TabularizedDnn(Callable):
    _table: ArrayNd
    _grid: hj_reachability.Grid

    @classmethod
    def from_dnn_and_grid(cls, dnn: Callable, grid: hj_reachability.Grid) -> 'TabularizedDnn':
        table = tabularize_dnn(dnn, grid)
        return cls(table, grid)

    def __call__(self, state: Vector) -> Vector:
        index = snap_state_to_grid_index(state, self._grid)
        controls = self._table[index].reshape((self._table.shape[-1], 1))
        return controls


def load_tabularized_ppo(grid: hj_reachability.Grid) -> TabularizedDnn:
    return TabularizedDnn.from_dnn_and_grid(load_policy_ppo(), grid)


def load_tabularized_sac(grid: hj_reachability.Grid) -> TabularizedDnn:
    return TabularizedDnn.from_dnn_and_grid(load_policy_sac(), grid)
