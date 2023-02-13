import os
import warnings

import hj_reachability
import matplotlib
from jax import numpy as jnp
from matplotlib import pyplot as plt

from refineNCBF.dynamic_systems.implementations.active_cruise_control import ActiveCruiseControlJAX, simplified_active_cruise_control_params
from refineNCBF.dynamic_systems.implementations.quadcopter import quadcopter_vertical_jax_hj
from refineNCBF.refining.hj_reachability_interface.hj_dynamics import HJControlAffineDynamics, ActorModes
from refineNCBF.refining.local_hjr_solver.solve import LocalHjrSolver
from refineNCBF.utils.files import visuals_data_directory, generate_unique_filename
from refineNCBF.utils.sets import compute_signed_distance
from refineNCBF.utils.visuals import ArraySlice2D, DimName

warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use("TkAgg")


def demo_local_hjr_classic_solver_on_active_cruise_control(save_result: bool = False):
    dynamics = quadcopter_vertical_jax_hj

    grid = hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj_reachability.sets.Box(
            [0, -8, -jnp.pi, -10],
            [10, 8, jnp.pi, 10]
        ),
        shape=(31, 25, 41, 25)
    )

    # define reach and avoid targets
    avoid_set = (
            (grid.states[..., 0] < 1)
            |
            (grid.states[..., 0] > 9)
    )

    reach_set = jnp.zeros_like(avoid_set, dtype=bool)

    terminal_values = compute_signed_distance(~avoid_set)

    solver = LocalHjrSolver.as_local_solver(
        dynamics=dynamics,
        grid=grid,
        avoid_set=avoid_set,
        reach_set=reach_set,
        terminal_values=terminal_values,
        max_iterations=100,
        verbose=True
    )

    initial_values = terminal_values.copy()
    active_set = ~avoid_set

    result = solver(active_set=active_set, initial_values=initial_values)

    if save_result:
        result.save(generate_unique_filename('data/local_update_results/result_qv_local', 'dill'))

    return result


if __name__ == '__main__':
    demo_local_hjr_classic_solver_on_active_cruise_control(save_result=True)
