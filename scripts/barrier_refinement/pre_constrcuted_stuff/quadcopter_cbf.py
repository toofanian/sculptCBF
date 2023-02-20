import hj_reachability
import numpy as np
import stable_baselines3
import torch
from gym import spaces

from refineNCBF.training.dnn_models.cbf import Cbf, NnCertifiedDict
from refineNCBF.training.dnn_models.stable_baselines_interface import StableBaselinesCallable
from refineNCBF.training.dnn_models.standardizer import Standardizer
from refineNCBF.training.dnn_models.tabularized_dnn import TabularizedDnn
from refineNCBF.utils.files import construct_full_path, FilePathRelative
from refineNCBF.utils.types import VectorBatch


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


def load_uncertified_states(certified_dict: NnCertifiedDict, standardizer: Standardizer) -> VectorBatch:
    uncertified_states = certified_dict['uns']
    violated_states = certified_dict['vio']
    total_states = uncertified_states + violated_states
    total_states_destandardized = standardizer.destandardize(np.array(total_states))
    return total_states_destandardized


def load_certified_states(certified_dict: NnCertifiedDict, standardizer: Standardizer) -> VectorBatch:
    certified_states = certified_dict['cert']
    total_states_destandardized = standardizer.destandardize(np.array(certified_states))
    return total_states_destandardized


def load_policy_sac(relative_path: FilePathRelative) -> StableBaselinesCallable:
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
        stable_baselines3.SAC.load(construct_full_path(relative_path), custom_objects=custom_objects)
    )


def load_tabularized_sac(grid: hj_reachability.Grid, relative_path: FilePathRelative) -> TabularizedDnn:
    return TabularizedDnn.from_dnn_and_grid(load_policy_sac(relative_path), grid)
