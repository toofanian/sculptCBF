import warnings

import hj_reachability
from jax import numpy as jnp

from refineNCBF.dynamic_systems.implementations.active_cruise_control_odp import ActiveCruiseControlOdp
from refineNCBF.refining.local_hjr_solver.solver_odp import create_global_solver_odp
from refineNCBF.utils.files import generate_unique_filename
from refineNCBF.utils.sets import compute_signed_distance

warnings.simplefilter(action='ignore', category=FutureWarning)


def wip_acc_global_odp(save_result: bool = False):
    dynamics = ActiveCruiseControlOdp()

    grid = hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj_reachability.sets.Box(
            [0, -20, 20],
            [1e3, 20, 80]
        ),
        shape=(3, 501, 501)
    )

    avoid_set = (
            (grid.states[..., 2] > 60)
            |
            (grid.states[..., 2] < 40)
    )

    reach_set = jnp.zeros_like(avoid_set, dtype=bool)

    terminal_values = compute_signed_distance(~avoid_set)

    solver = create_global_solver_odp(
        dynamics=dynamics,
        grid=grid,
        avoid_set=avoid_set,
        reach_set=reach_set,
        terminal_values=terminal_values,
        max_iterations=70,
        solver_timestep=-.1,
        verbose=True
    )

    initial_values = terminal_values.copy()
    active_set = jnp.ones_like(initial_values, dtype=bool)

    result = solver(active_set=active_set, initial_values=initial_values)

    if save_result:
        result.save(generate_unique_filename('data/local_update_results/wip_acc_global_odp', 'dill'))

    return result


if __name__ == '__main__':
    wip_acc_global_odp(save_result=True)
