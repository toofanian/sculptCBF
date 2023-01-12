# Barrier
import json

import numpy as np
import jax.numpy as jnp
import torch

from refineNCBF.training.dnn_models.cbf import Cbf
from refineNCBF.utils.files import construct_full_path, FilePathRelative
from refineNCBF.training.dnn_models.standardizer import Standardizer
from refineNCBF.utils.types import VectorBatch, MaskNd


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


def load_uncertified_states():
    with open(construct_full_path('data/trained_NCBFs/quad4d_boundary/quad4d_boundary_cert_results.json')) as f:
        quad4d_result_dict = json.load(f)
    uncertified_states = quad4d_result_dict['uns']
    violated_states = quad4d_result_dict['vio']
    total_states = uncertified_states + violated_states

    standardizer = load_standardizer()
    total_states_destandardized = standardizer.destandardize(total_states)
    return tuple(map(tuple, total_states_destandardized))


def load_uncertified_states_np() -> VectorBatch:
    with open(construct_full_path('data/trained_NCBFs/quad4d_boundary/quad4d_boundary_cert_results.json')) as f:
        quad4d_result_dict = json.load(f)
    uncertified_states = quad4d_result_dict['uns']
    violated_states = quad4d_result_dict['vio']
    total_states = uncertified_states + violated_states

    standardizer = load_standardizer()
    total_states_destandardized = standardizer.destandardize(np.array(total_states))
    return total_states_destandardized


def load_uncertified_mask(path: FilePathRelative) -> MaskNd:
    return jnp.load(construct_full_path(path))
