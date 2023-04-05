import numpy as np
from refineNCBF.neural_barrier_interface.load_interface import load_cbf
from refineNCBF.neural_barrier_interface.neural_cbf import NeuralControlAffineCBF
from refineNCBF.utils.emperical_validation import validate_cbf, test_nominal_controller
from refineNCBF.local_hjr_solver.result import LocalUpdateResult
from refine_cbfs import TabularControlAffineCBF
from cbf_opt import ControlAffineDynamics
from cbf_opt import utils as cbf_utils
from scripts.quad4d.learned_cbf import quad4d_learned_barrier_params
from refineNCBF.neural_barrier_interface.load_interface import load_policy_sac

import logging

logger = logging.getLogger(__name__)

# Surpress warnings from logger
logging.getLogger("cbf_opt.asif").setLevel(logging.ERROR)


class QuadVerticalDynamics(ControlAffineDynamics):
    STATES = ["Y", "YDOT", "PHI", "PHIDOT"]
    CONTROLS = ["T1", "T2"]
    PERIODIC_DIMS = [2]

    def __init__(self, params, **kwargs):
        self.Cd_v = params["Cd_v"]
        self.g = params["g"]
        self.Cd_phi = params["Cd_phi"]
        self.mass = params["mass"]
        self.length = params["length"]
        self.Iyy = params["Iyy"]
        super().__init__(params, **kwargs)

    def open_loop_dynamics(self, state: np.ndarray, time: float = 0) -> np.ndarray:
        f = np.zeros_like(state)
        f[..., 0] = state[..., 1]
        f[..., 1] = -self.Cd_v / self.mass * state[..., 1] - self.g
        f[..., 2] = state[..., 3]
        f[..., 3] = -self.Cd_phi / self.Iyy * state[..., 3]
        return f

    def control_matrix(self, state: np.ndarray, time: float = 0) -> np.ndarray:
        B = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        B[..., 1, 0] = np.cos(state[..., 2]) / self.mass
        B[..., 1, 1] = np.cos(state[..., 2]) / self.mass
        B[..., 3, 0] = -self.length / self.Iyy
        B[..., 3, 1] = self.length / self.Iyy
        return B

    def disturbance_jacobian(self, state: np.ndarray, time: float = 0) -> np.ndarray:
        return np.repeat(np.zeros_like(state)[..., None], 1, axis=-1)

    def state_jacobian(self, state: np.ndarray, control: np.ndarray, time: float = 0) -> np.ndarray:
        J = np.repeat(np.zeros_like(state)[..., None], state.shape[-1], axis=-1)
        J[..., 0, 1] = 1.0
        J[..., 1, 1] = -self.Cd_v / self.mass
        J[..., 1, 2] = -(control[..., 0] + control[..., 1]) * np.sin(state[..., 2]) / self.mass
        J[..., 2, 3] = 1.0
        J[..., 3, 3] = -self.Cd_phi / self.Iyy
        return J


gravity: float = 9.81
mass: float = 2.5
Cd_v: float = 0.25
drag_coefficient_phi: float = 0.02255
length_between_copters: float = 1.0
moment_of_inertia: float = 1.0

u_min: float = 0
u_max: float = 0.75 * mass * gravity
dynamics = QuadVerticalDynamics(
    params={
        "Cd_v": Cd_v,
        "g": gravity,
        "Cd_phi": drag_coefficient_phi,
        "mass": mass,
        "length": length_between_copters,
        "Iyy": moment_of_inertia,
        "dt": 0.02,
    }
)


class NominalPolicyWrapper:
    def __init__(self, nominal_policy, requires_goal=False):
        self.nominal_policy = nominal_policy
        self.requires_goal = requires_goal


x_nom = np.array([5.0, 0.0, 0.0, 0.0])
u_nom = 0.5 * dynamics.mass * dynamics.g * np.ones(2)


cbf, standardizer, certified_dict = load_cbf()

neural_cbf = NeuralControlAffineCBF(dynamics, {}, V_nn=cbf, normalizer=standardizer)


local_hjr_dict = {
    "global_jax": "_20230307_191036",
    # "local_jax": "20230307_025911",
    # "local_odp": "20230307_180336",
    # "global_odp": "_20230307_183541",
    "local_odp_finegrid": "cbf_20230320_043531",
    "local_odp_finegrid_sdf": "sdf_20230319_093709",
}


cbfs = {}

for i, (key, value) in enumerate(local_hjr_dict.items()):
    try:
        patching_result = LocalUpdateResult.load("data/local_update_results/{}.dill".format(value))
    except FileNotFoundError:
        patching_result = LocalUpdateResult.load("data/quad4d/{}.dill".format(value))
    if i == 0:
        grid = patching_result.grid

    result_cbf_values = patching_result.iterations[-1].computed_values
    converged_cbvf = TabularControlAffineCBF(dynamics, {}, grid=patching_result.grid)
    converged_cbvf.vf_table = result_cbf_values
    cbfs[key] = converged_cbvf


cbfs["neural"] = neural_cbf


A, B = dynamics.linearized_ct_dynamics(x_nom, u_nom)

A_d, B_d = dynamics.linearized_dt_dynamics(x_nom, u_nom)

Q = np.diag([1, 0.1, 1.0, 1.0])
R = np.eye(2)

K = cbf_utils.lqr(A_d, B_d, Q, R)

A_cl = A - B @ K
assert (np.linalg.eig(A_cl)[0] < 0).all(), "System is not stable"


lqr_controller = lambda u_ref, F: lambda x_ref: lambda x, t: np.atleast_2d(
    np.clip(u_ref - (F @ (x - x_ref).T).T, u_min * np.ones(2), u_max * np.ones(2))
)


lqr_policy = lqr_controller(u_nom, K)


lqr_nominal_policy = NominalPolicyWrapper(lqr_policy, requires_goal=True)


neural_policy = load_policy_sac(
    control_bounds=np.array([u_min * np.ones(2), u_max * np.ones(2)]),
)

neural_policy_with_time = lambda x, t: neural_policy(x)

neural_policy_wrapped = NominalPolicyWrapper(neural_policy_with_time, requires_goal=False)

nominal_policies_dict = {
    "neural": neural_policy_wrapped,
    "lqr": lqr_nominal_policy,
}

from datetime import datetime

file_name = datetime.now().strftime("%y%m%d_%H%M")

df = validate_cbf(
    grid,
    cbfs,
    nominal_policies_dict,
    nbr_samples=100,
    control_bounds=np.array([u_min * np.ones(2), u_max * np.ones(2)]),
    all_cbfs_positive=False,
)
