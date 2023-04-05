import numpy as np
from refineNCBF.utils.types import *
import hj_reachability
from typing import Callable, Optional
from cbf_opt import ControlAffineASIF, Dynamics, CBF
from experiment_wrapper import RolloutTrajectory
import pandas as pd


def validate_cbf(
    grid: hj_reachability.Grid,
    cbfs: Dict[str, CBF],
    nominal_policies: Dict[str, Callable[[ArrayNd], ArrayNd]],
    control_bounds: ArrayNd = [-np.inf, np.inf],
    nbr_samples: int = 1000,
    random_seed: Optional[int] = None,
    all_cbfs_positive: bool = True,
    curr_df: Optional[pd.DataFrame] = None,
    verbose: bool = False,
):
    """
    Samples nbr_samples states with value > 0 and rolls out a trajectory from each state for 100 steps to a random goal in the state space."""
    if random_seed is not None:
        random_state = np.random.RandomState(random_seed)
    else:
        random_state = np.random.RandomState(1)

    if not isinstance(cbfs, dict):
        cbfs = {"cbf": cbfs}
    if not isinstance(nominal_policies, dict):
        nominal_policies = {"nominal_policy": nominal_policies}
    benchmark_cbf = cbfs[list(cbfs.keys())[0]]
    dynamics = benchmark_cbf.dynamics
    starting_states = None
    i = 0
    while starting_states is None or len(starting_states) < nbr_samples:

        # Sample nbr_samples states from the grid
        states = (np.array(grid.domain.hi) - np.array(grid.domain.lo)) * random_state.rand(
            nbr_samples, grid.ndim
        ) + np.array(grid.domain.lo)

        if all_cbfs_positive:
            # Filter out states with value <= 0
            states = states[
                np.all(
                    [cbf.vf(states, time=0.0) > 0.0 for cbf in cbfs.values()],
                    axis=0,
                )
            ]
        else:
            # Valid comparison when 1st CBF in list has "smallest" safe set
            states = states[benchmark_cbf.vf(states, time=0.0) > 0.0]
        if starting_states is None:
            starting_states = states
        else:
            starting_states = np.vstack((starting_states, states))
        if verbose:
            print("Iteration {}, starting states size: {}".format(i, len(starting_states)))
        i += 1

    starting_states = np.array(starting_states)[:nbr_samples]
    # Find a random goal
    goal = (np.array(grid.domain.hi) - np.array(grid.domain.lo)) * random_state.rand(nbr_samples, grid.ndim) + np.array(
        grid.domain.lo
    )

    # We want the goal to not be something crazy
    if dynamics.n_dims == 4:
        goal[..., 1:] = 0.0
    elif dynamics.n_dims == 6:
        goal[..., 1] = 0.0
        goal[..., 3:] = 0.0
    # Rollout trajectories
    alpha_low = lambda x: 0.5 * x
    from datetime import datetime

    file_name = datetime.now().strftime("%y%m%d_%H%M")

    safe_controllers_dict = {}
    for nominal_policy_name in nominal_policies:
        nominal_policy = nominal_policies[nominal_policy_name]
        if nominal_policy.requires_goal == True:
            nom_policy = nominal_policy.nominal_policy(goal)
        else:
            nom_policy = nominal_policy.nominal_policy
        assert nom_policy(np.random.rand(nbr_samples, dynamics.n_dims), 0.0).shape == (
            nbr_samples,
            dynamics.control_dims,
        )
        if curr_df is not None and nominal_policy_name in curr_df.controller.unique():
            print(f"Skipping {nominal_policy_name} because it already exists in the dataframe")
        else:
            safe_controllers_dict[nominal_policy_name] = nom_policy
        for cbf_inst_name in cbfs:
            combined_name = cbf_inst_name + nominal_policy_name
            cbf_inst = cbfs[cbf_inst_name]
            asif = ControlAffineASIF(
                dynamics,
                cbf_inst,
                nominal_policy=nom_policy,
                alpha=alpha_low,
                umin=control_bounds[0],
                umax=control_bounds[1],
                test=False,
            )
            if curr_df is not None and combined_name in curr_df.controller.unique():
                print(f"Skipping {combined_name} because it already exists in the dataframe")
                continue
            safe_controllers_dict[combined_name] = asif

    experiment = RolloutTrajectory(
        "Validation", start_x=starting_states, t_sim=10.0, n_sims_per_start=1, save_location=file_name
    )
    df = experiment.run(dynamics, safe_controllers_dict, control_bounds=control_bounds)
    return df
