import numpy as np
from refineNCBF.neural_barrier_interface.neural_cbf import NeuralControlAffineCBF
from refineNCBF.utils.emperical_validation import validate_cbf, test_nominal_controller
from refineNCBF.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.utils.files import construct_refine_ncbf_path
from refine_cbfs import TabularControlAffineCBF
from cbf_opt import ControlAffineDynamics
from cbf_opt import utils as cbf_utils
from quad6d.learned_cbf import load_cbf, load_policy_sac
import torch
import logging
from dynamics_interface import QuadPlanarDynamicsInterface

logger = logging.getLogger(__name__)
dynamics_interface = QuadPlanarDynamicsInterface()
dynamics = dynamics_interface.dynamics
# Surpress warnings from logger
logging.getLogger("cbf_opt.asif").setLevel(logging.ERROR)


class NominalPolicyWrapper:
    def __init__(self, nominal_policy, requires_goal=False):
        self.nominal_policy = nominal_policy
        self.requires_goal = requires_goal


x_nom = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
u_nom = 0.5 * dynamics.mass * dynamics.g * np.ones(2)


cbf, standardizer, certified_dict = load_cbf()

neural_cbf = NeuralControlAffineCBF(dynamics, {}, V_nn=cbf, normalizer=standardizer)


local_hjr_dict = {
    # "global_jax": "_20230307_191036",
    # # "local_jax": "20230307_025911",
    # # "local_odp": "20230307_180336",
    # # "global_odp": "_20230307_183541",
    # "local_odp_finegrid": "cbf_20230320_043531",
    # "local_odp_finegrid_sdf": "sdf_20230319_093709",
    "global_odp": "_20230322_233200",
    # "global_odp_w_sdf": "_20230328_042741"
    "local_odp": "_20230322_175659",
    # "local_odp_sdf": "_20230324_234357",
    # "local_odp_nosdf": "_20230324_232335",
    # "local_odp_nomins": "_20230322_185034",
}

local_conv_itr_dict = {"global_odp": 19, "local_odp": 4}


cbfs = {}

for i, (key, value) in enumerate(local_hjr_dict.items()):
    try:
        patching_result = LocalUpdateResult.load("data/local_update_results/{}.dill".format(value))
    except FileNotFoundError:
        try:
            patching_result = LocalUpdateResult.load("data/6dim/{}.dill".format(value))
        except FileNotFoundError:
            try:
                patching_result = LocalUpdateResult.load("/data/6dim/{}.dill".format(value))
            except FileNotFoundError:
                patching_result = LocalUpdateResult.load("/data_extension/6dim/{}.dill".format(value))
    if i == 0:
        grid = patching_result.grid
    # else:
    # assert (grid.states == patching_result.grid.states).all()
    result_cbf_values = patching_result.iterations[-1].computed_values
    total_len = len(result_cbf_values)
    half_result_cbf_values = patching_result.iterations[int(total_len / 2)].computed_values
    quarter_result_cbf_values = patching_result.iterations[int(total_len / 4)].computed_values
    eight_result_cbf_values = patching_result.iterations[int(total_len / 8)].computed_values

    converged_result_cbf_values = patching_result.iterations[local_conv_itr_dict[key]].computed_values
    converged_cbvf = TabularControlAffineCBF(dynamics, {}, grid=patching_result.grid)
    converged_cbvf.vf_table = converged_result_cbf_values
    cbfs[key] = converged_cbvf


cbfs["neural"] = neural_cbf


A, B = dynamics.linearized_ct_dynamics(x_nom, u_nom)

A_d, B_d = dynamics.linearized_dt_dynamics(x_nom, u_nom)

Q = np.diag([1, 0.1, 1.0, 0.1, 1.0, 1.0])
R = np.eye(2)

K = cbf_utils.lqr(A_d, B_d, Q, R)

A_cl = A - B @ K
assert (np.linalg.eig(A_cl)[0] < 0).all(), "System is not stable"


lqr_controller = lambda u_ref, F: lambda x_ref: lambda x, t: np.atleast_2d(
    np.clip(u_ref - (F @ (x - x_ref).T).T, dynamics_interface.u_min * np.ones(2), dynamics_interface.u_max * np.ones(2))
)


lqr_policy = lqr_controller(u_nom, K)


lqr_nominal_policy = NominalPolicyWrapper(lqr_policy, requires_goal=True)


neural_policy = load_policy_sac(
    construct_refine_ncbf_path("data/quad6d/learned_policy.zip"),
    control_bounds=np.array([dynamics_interface.u_min * np.ones(2), dynamics_interface.u_max * np.ones(2)]),
)

neural_policy_with_time = lambda x, t: neural_policy(x)

neural_policy_wrapped = NominalPolicyWrapper(neural_policy_with_time, requires_goal=False)

nominal_policies_dict = {
    "neural": neural_policy_wrapped,
    "lqr": lqr_nominal_policy,
}

from datetime import datetime

file_name = datetime.now().strftime("%y%m%d_%H%M")

import pandas as pd

# curr_df = pd.read_csv("230317_0644.csv")
# df = validate_cbf(grid, tab_cbf, nom_control, nbr_samples=10)
# df = validate_cbf(grid, second_tab_cbf, nom_control, nbr_samples=10)
df = validate_cbf(
    grid,
    cbfs,
    nominal_policies_dict,
    nbr_samples=1000,
    control_bounds=np.array([dynamics_interface.u_min * np.ones(2), dynamics_interface.u_max * np.ones(2)]),
    all_cbfs_positive=False
    # curr_df=curr_df,
)
# Get name for file in following format "230315_1234"
# df.to_csv("{}_cbf_emperical.csv".format(file_name))
# from experiment_wrapper import StateSpaceExperiment

# state_space_rollout = StateSpaceExperiment("Rollout", start_x=0, x_indices=[2, 0])
# fig_handle = state_space_rollout.plot(dynamics, df, add_direction=False)

# # Get current file
# fig_handle[0][1].savefig("rollout_cbf_init.png")
# df = test_nominal_controller(grid, dynamics, nom_control)

# from experiment_wrapper import StateSpaceExperiment

# state_space_rollout = StateSpaceExperiment("Rollout", start_x=0, x_indices=[0, 2])
# fig_handle = state_space_rollout.plot(dynamics, df, add_direction=False)

# # Get current file
# fig_handle[0][1].savefig("rollout.png")
