from typing import Union, Dict, List

import jax
import numpy as np
import torch

ArrayNd = Union[np.ndarray, torch.Tensor, jax.Array]
Scalar = ArrayNd
Vector = ArrayNd
Matrix = ArrayNd
ScalarBatch = ArrayNd
VectorBatch = ArrayNd  # array of shape (batch, dims)
MatrixBatch = ArrayNd  # array of shape (batch, dims, dims)
MaskNd = ArrayNd  # array of dimension N, dtype=bool

NnCertifiedDict = Dict[str, List]
