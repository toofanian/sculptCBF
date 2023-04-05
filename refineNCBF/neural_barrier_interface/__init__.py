from typing import Tuple, Union
import attr
import numpy as np


@attr.dataclass
class LearnedBarrierParams:
    input_dim: int
    hidden_dim: int
    barrier_path: str
    cellwidths: Union[float, Tuple[float]]

    observation_limits: Tuple[np.ndarray, np.ndarray]
    control_limits: Tuple[np.ndarray, np.ndarray]
    policy_path: str
