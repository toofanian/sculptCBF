# Barrier
import torch

from refineNCBF.training.dnn_models.cbf import Cbf
from refineNCBF.utils.files import construct_full_path


def load_quadcopter_cbf() -> Cbf:
    device = 'cpu'
    print("loading CBF model...")
    cbf_func = Cbf(4, 256)
    cbf_ckpt = torch.load(construct_full_path('data/trained_NCBFs/quad4d_cbf.pth'), map_location=device)
    cbf_func.load_state_dict(cbf_ckpt['model_state_dict'])
    cbf_func.to(device)
    return cbf_func
