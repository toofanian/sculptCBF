import warnings

import matplotlib
from jax import numpy as jnp

import hj_reachability
from refineNCBF.dynamic_systems.quadcopter import quadcopter_vertical_jax_hj
from refineNCBF.local_hjr_solver.solve import LocalHjrSolver
from refineNCBF.utils.files import generate_unique_filename
from refineNCBF.utils.sets import compute_signed_distance

warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use("TkAgg")


def demo_local_hjr_classic_solver_on_active_cruise_control(save_result: bool = False):
    dynamics = quadcopter_vertical_jax_hj

    grid = hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj_reachability.sets.Box(
            [0, -8, -jnp.pi, -10],
            [10, 8, jnp.pi, 10]
        ),
        shape=(75, 41, 75, 41)
    )

    # define reach and avoid targets
    avoid_set = (
            (grid.states[..., 0] < 1)
            |
            (grid.states[..., 0] > 9)
    )

    reach_set = jnp.zeros_like(avoid_set, dtype=bool)

    terminal_values = compute_signed_distance(~avoid_set)

    solver = LocalHjrSolver.as_global_solver(
        dynamics=dynamics,
        grid=grid,
        avoid_set=avoid_set,
        reach_set=reach_set,
        terminal_values=terminal_values,
        max_iterations=100,
        change_fraction=.999,
        verbose=True,
    )

    initial_values = terminal_values.copy()
    active_set = jnp.ones_like(avoid_set, dtype=bool)

    result = solver(active_set=active_set, initial_values=initial_values)

    if save_result:
        result.save(generate_unique_filename('data/local_update_results/result_qv_global', 'dill'))

    return result


if __name__ == '__main__':
    demo_local_hjr_classic_solver_on_active_cruise_control(save_result=True)
