from refineNCBF.neural_barrier_interface import LearnedBarrierParams
import numpy as np

# Add the current directory to the path so that we can import the dynamics
import sys

sys.path.append(".")
from dynamics import default_quadcopter_vertical_params, QuadcopterVertical


dyn_sys = QuadcopterVertical.from_specs(default_quadcopter_vertical_params)


quad4d_learned_barrier_params = LearnedBarrierParams(
    input_dim=4,
    hidden_dim=512,
    barrier_path="data/quad4d/learned_barrier",
    cellwidths=0.02551,
    observation_limits=(np.array([0.0, -8.0, -np.pi, -10.0]), np.array([10.0, 8.0, np.pi, 10.0])),
    control_limits=(dyn_sys.control_lower_bounds, dyn_sys.control_upper_bounds),
    policy_path="data/quad4d/learned_policy.zip",
)
