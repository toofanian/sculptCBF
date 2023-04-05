import os
import attr
import hj_reachability
import numpy as np
from refineNCBF.neural_barrier_interface import LearnedBarrierParams
from refineNCBF.utils.tables import TabularizedDnn
import stable_baselines3
from gym import spaces
from refineNCBF.neural_barrier_interface.stable_baselines_interface import StableBaselinesCallable
import torch
from neural_barrier_kinematic_model.cbf_tanh_2_layer import CBFTanh2Layer
from neural_barrier_kinematic_model.standardizer import Standardizer
from refineNCBF.utils.files import construct_refine_ncbf_path
from refineNCBF.utils.types import NnCertifiedDict
from refineNCBF.utils.files import construct_refine_ncbf_path

import json
from typing import Tuple, Union


def load_cbf(params: LearnedBarrierParams) -> Tuple[CBFTanh2Layer, Standardizer, NnCertifiedDict]:
    device = "cpu"
    cbf = CBFTanh2Layer(params.input_dim, params.hidden_dim)
    cbf_ckpt = torch.load(
        os.path.join(construct_refine_ncbf_path(params.barrier_path), "barrier.pth"), map_location=device
    )
    cbf.load_state_dict(cbf_ckpt["model_state_dict"])
    cbf.to(device)

    standardizer = Standardizer(fp=os.path.join(construct_refine_ncbf_path(params.barrier_path), "standardizer.npy"))
    standardizer.initialize_from_file()

    with open(os.path.join(construct_refine_ncbf_path(params.barrier_path), "cert_results.json")) as f:
        certified_dict: NnCertifiedDict = json.load(f)

    return cbf, standardizer, certified_dict


def load_policy_sac(params: LearnedBarrierParams) -> StableBaselinesCallable:
    observation_space = spaces.Box(
        low=params.observation_limits[0],
        high=params.observation_limits[1],
        shape=(len(params.observation_limits[0]),),
        dtype=np.float64,
    )
    action_space = spaces.Box(
        low=params.control_limits[0],
        high=params.control_limits[1],
        shape=(len(params.control_limits[0]),),
        dtype=np.float64,
    )

    custom_objects = {
        "observation_space": observation_space,
        "action_space": action_space,
    }
    try:
        return StableBaselinesCallable(
            stable_baselines3.SAC.load(construct_refine_ncbf_path(params.policy_path), custom_objects=custom_objects)
        )
    except FileNotFoundError:
        return StableBaselinesCallable(stable_baselines3.SAC.load(params.policy_path, custom_objects=custom_objects))


def load_tabularized_sac(grid: hj_reachability.Grid, path: str) -> TabularizedDnn:
    return TabularizedDnn.from_dnn_and_grid(load_policy_sac(path), grid)
