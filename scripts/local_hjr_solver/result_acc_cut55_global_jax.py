import warnings

from jax import numpy as jnp

import hj_reachability
from refineNCBF.dynamic_systems.active_cruise_control import ActiveCruiseControlJAX, \
    simplified_active_cruise_control_params
from refineNCBF.hj_reachability_interface.hj_dynamics import HJControlAffineDynamics, ActorModes
from refineNCBF.local_hjr_solver.solve import LocalHjrSolver
from refineNCBF.utils.files import generate_unique_filename
from refineNCBF.utils.sets import compute_signed_distance

warnings.simplefilter(action='ignore', category=FutureWarning)


def acc_global_jax(save_result: bool = False):
    dynamics = HJControlAffineDynamics.from_parts(
        control_affine_dynamic_system=ActiveCruiseControlJAX.from_params(simplified_active_cruise_control_params),
        control_mode=ActorModes.MAX,
        disturbance_mode=ActorModes.MIN,
    )

    grid = hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj_reachability.sets.Box(
            [0, -20, 20],
            [1e3, 20, 80]
        ),
        shape=(3, 75, 75)
    )

    avoid_set = get_cut55_avoid()
    reach_set = jnp.zeros_like(avoid_set, dtype=bool)

    terminal_values = compute_signed_distance(~avoid_set)

    solver = LocalHjrSolver.as_global_solver(
        dynamics=dynamics,
        grid=grid,
        avoid_set=avoid_set,
        reach_set=reach_set,
        terminal_values=terminal_values,
        max_iterations=100,
        change_fraction=1,
        verbose=True
    )

    initial_values = terminal_values.copy()
    active_set = jnp.ones_like(avoid_set, dtype=bool)

    result = solver(active_set=active_set, initial_values=initial_values)

    if save_result:
        result.save(
            generate_unique_filename('data/local_update_results/wip_acc_cut55_global_jax',
                                     'dill'))

    return result


def get_cut55_avoid():
    dynamics = HJControlAffineDynamics.from_parts(
        control_affine_dynamic_system=ActiveCruiseControlJAX.from_params(simplified_active_cruise_control_params),
        control_mode=ActorModes.MAX,
        disturbance_mode=ActorModes.MIN,
    )

    grid = hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj_reachability.sets.Box(
            [0, -20, 20],
            [1e3, 20, 80]
        ),
        shape=(3, 75, 75)
    )

    avoid_set = (
            (grid.states[..., 2] > 60)
            |
            (grid.states[..., 2] < 40)
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
        change_fraction=1,
        verbose=False
    )

    initial_values = terminal_values.copy()
    active_set = jnp.ones_like(avoid_set, dtype=bool)

    result = solver(active_set=active_set, initial_values=initial_values)

    cut55 = result.get_viability_kernel() & (grid.states[..., 2] < 55)

    return ~cut55


if __name__ == '__main__':
    acc_global_jax(save_result=True)
