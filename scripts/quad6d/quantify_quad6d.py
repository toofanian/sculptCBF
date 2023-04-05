from cbf_opt.dynamics import ControlAffineDynamics
import numpy as np
import pandas as pd
from experiment_wrapper import RolloutTrajectory
import warnings
from dynamics_interface import QuadPlanarDynamicsInterface

dynamics_interface = QuadPlanarDynamicsInterface()
dynamics = dynamics_interface.dynamics

# Hide pandas warnings
# warnings.filterwarnings("ignore", category=pd)


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
