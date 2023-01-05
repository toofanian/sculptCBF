from typing import Union

import jax
import numpy as np
import torch

ArrayNd = Union[np.ndarray, torch.Tensor, jax.Array]
ScalarBatch = ArrayNd
VectorBatch = ArrayNd  # array of shape (batch, dims)
MatrixBatch = ArrayNd  # array of shape (batch, dims, dims)
MaskNd = ArrayNd  # array of dimension N, dtype=bool
