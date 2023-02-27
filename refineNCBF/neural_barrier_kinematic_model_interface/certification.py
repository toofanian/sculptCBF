import numpy as np

from neural_barrier_kinematic_model.standardizer import Standardizer
from refineNCBF.utils.types import VectorBatch, NnCertifiedDict


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
