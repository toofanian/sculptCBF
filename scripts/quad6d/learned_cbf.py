from neural_barrier_kinematic_model import LearnedBarrierParams
import numpy as np
from refineNCBF.dynamic_systems.planar_quadcopter import default_quadcopter_planar_params, QuadcopterPlanar


dyn_sys = QuadcopterPlanar.from_specs(default_quadcopter_planar_params)


quad4d_learned_barrier_params = LearnedBarrierParams(
    input_dim=6,
    hidden_dim=1024,
    barrier_path="data/quad6d/learned_barrier",
    cellwidths=0.03125,
    observation_limits=(
        np.array([-7.94, -15.91, -0.82, -6.13, -3.14, -9.35]),
        np.array([7.93, 14.98, 11.34, 10.03, 3.14, 8.19]),
    ),
    control_limits=(dyn_sys.control_lower_bounds, dyn_sys.control_upper_bounds),
    policy_path="data/quad6d/learned_policy.zip",
)
