from typing import Dict, List

import torch
import torch.nn as nn

from refineNCBF.utils.types import Vector


class Cbf(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Cbf, self).__init__()
        self.affine1 = nn.Linear(input_dim, hidden_dim)
        self.affine2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.affine1(x)
        x = torch.tanh(x)
        x = self.affine2(x)
        return x


NnCertifiedDict = Dict[str, List[Vector]]
