from cbf_opt.dynamics import ControlAffineDynamics
import numpy as np
import pandas as pd
from experiment_wrapper import RolloutTrajectory
import warnings

# Hide pandas warnings
# warnings.filterwarnings("ignore", category=pd)


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


class EmpericalSafety(RolloutTrajectory):
    def quantify(self, dynamics, results_df: pd.DataFrame, **kwargs):
        print(
            "Quantifying safety of rollouts for {}, state_dim {}, control dim {}".format(
                dynamics.__class__.__name__,
                dynamics.n_dims,
                dynamics.control_dims,
            )
        )
        print(
            "Experiment general statistics:\t Duration: {} seconds, Controller dt: {},\t Number of rollouts: {}".format(
                results_df.t.max() + dynamics.dt, dynamics.dt, len(results_df.rollout.unique())
            )
        )
        for controller in results_df.controller.unique():
            curr_df = results_df[results_df.controller == controller]
            mask1 = curr_df.measurement.isin(["T1", "T2"])
            mask2 = (curr_df.value >= 0.0) & (curr_df.value <= 19)
            if ~((mask1 & mask2).sum() == (mask1).sum()):
                print("Controller {} has invalid control inputs".format(controller))
            unsafe_cells = curr_df[curr_df.unsafe == True]
            unsafe_cell_count = unsafe_cells.shape[0]
            safe_cells = curr_df[curr_df.unsafe == False]
            safe_cell_count = safe_cells.shape[0]
            if unsafe_cell_count == 0 and safe_cell_count == 0:
                continue
            else:
                print(
                    "controller: {}, share of unsafe states (agnostic of different trajectories): {:.2f} %".format(
                        controller, unsafe_cell_count / (unsafe_cell_count + safe_cell_count) * 100
                    ),
                    end="\t",
                )

                nbr_unsafe_rollouts = unsafe_cells.rollout.unique().shape[0]
                nbr_safe_rollouts = safe_cells.rollout.unique().shape[0]
                nbr_rollouts = len(results_df.rollout.unique())

                print(
                    "Share of unsafe trajectories (with at least 1 timestep unsafe): {:.2f} %".format(
                        nbr_unsafe_rollouts / nbr_rollouts * 100
                    )
                )


# file_name = "230316_0012_cbf_emperical"
# file_name = "230316_0055_cbf_emperical"
# file_name = "230316_2131_cbf_emperical"
# file_name = "230317_0032"

# file_names = ["230316_0055_cbf_emperical", "230316_2131_cbf_emperical"]
# file_names += ["230317_0644_addt", "230317_1620"]
import os


def main(args):
    # Get directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for file_name in args.file_names:
        print(file_name)
        df = pd.read_csv(os.path.join(dir_path, "{}.csv".format(file_name)), low_memory=False)
        evalu = EmpericalSafety("test", 0)
        evalu.quantify(dynamics, df)


if __name__ == "__main__":
    import argparse

    # Parse argument that is called --file_names
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_names", nargs="+", help="List of file names to evaluate")
    args = parser.parse_args()
    main(args)
