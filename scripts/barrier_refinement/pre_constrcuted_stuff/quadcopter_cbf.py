import json
from typing import Callable, Union

import attr
import hj_reachability
import numpy as np
import stable_baselines3
import torch
from gym import spaces

from refineNCBF.training.dnn_models.cbf import Cbf
from refineNCBF.training.dnn_models.standardizer import Standardizer
from refineNCBF.utils.files import construct_full_path
from refineNCBF.utils.tables import tabularize_dnn, snap_state_to_grid_index
from refineNCBF.utils.types import VectorBatch, ArrayNd, Vector


def load_quadcopter_cbf() -> Cbf:
    device = 'cpu'
    print("loading CBF model...")
    cbf_func = Cbf(4, 256)
    cbf_ckpt = torch.load(construct_full_path('data/trained_NCBFs/sac_policy/quad4d_sac_cbf.pth'), map_location=device)
    cbf_func.load_state_dict(cbf_ckpt['model_state_dict'])
    cbf_func.to(device)
    return cbf_func


def load_standardizer() -> Standardizer:
    standardizer = Standardizer(fp=construct_full_path('data/trained_NCBFs/sac_policy/quad4d_sac_standardizer.npy'))
    standardizer.initialize_from_file()
    return standardizer


def load_uncertified_states() -> VectorBatch:
    with open(construct_full_path('data/trained_NCBFs/sac_policy/quad4d_sac_boundary_cert_results.json')) as f:
        quad4d_result_dict = json.load(f)
    uncertified_states = quad4d_result_dict['uns']
    violated_states = quad4d_result_dict['vio']
    total_states = uncertified_states + violated_states

    standardizer = load_standardizer()
    total_states_destandardized = standardizer.destandardize(np.array(total_states))
    return total_states_destandardized


def load_certified_states() -> VectorBatch:
    with open(construct_full_path('data/trained_NCBFs/sac_policy/quad4d_sac_boundary_cert_results.json')) as f:
        quad4d_result_dict = json.load(f)
    certified_states = quad4d_result_dict['cert']
    standardizer = load_standardizer()
    total_states_destandardized = standardizer.destandardize(np.array(certified_states))
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
    low = np.array([0., -8., -np.pi, -10.])
    high = np.array([10., 8., np.pi, 10.])
    observation_space = spaces.Box(
        low=low,
        high=high,
        shape=(len(low),),
        dtype=np.float64,
    )

    ac_high = np.array([20, 20])
    action_space = spaces.Box(
        low=-ac_high,
        high=ac_high,
        shape=(len(ac_high),),
        dtype=np.float64,
    )

    custom_objects = {
        "observation_space": observation_space,
        "action_space": action_space,
    }

    return StableBaselinesCallable(
        stable_baselines3.SAC.load(construct_full_path('data/trained_NCBFs/sac_policy/best_model-sac.zip'), custom_objects=custom_objects)
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
