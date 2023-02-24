import warnings

import matplotlib
from jax import numpy as jnp

import hj_reachability
from refineNCBF.dynamic_systems.implementations.quadcopter_fixed_policy import load_quadcopter_sac_jax_hj
from refineNCBF.refining.local_hjr_solver.solve import LocalHjrSolver
from refineNCBF.utils.files import generate_unique_filename
from refineNCBF.utils.sets import compute_signed_distance

warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use('TkAgg')


def result_qv_sac_global(save_result: bool = False):
    grid = hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj_reachability.sets.Box(
            [-1, -8, -jnp.pi, -10],
            [11, 8, jnp.pi, 10]
        ),
        shape=(101, 101, 101, 101)
    )

    dynamics = load_quadcopter_sac_jax_hj(grid=grid, relative_path='data/trained_NCBFs/feb18/best_model-3.zip')

    avoid_set = (
            (grid.states[..., 0] < 0)
            |
            (grid.states[..., 0] > 10)
    )
    reach_set = jnp.zeros_like(avoid_set, dtype=bool)

    terminal_values = compute_signed_distance(~avoid_set)

    solver = LocalHjrSolver.as_marching_solver(
        dynamics=dynamics,
        grid=grid,
        avoid_set=avoid_set,
        reach_set=reach_set,
        terminal_values=terminal_values,
        max_iterations=30,
        verbose=True,
    )

    initial_values = terminal_values.copy()
    # active_set = (
    #         get_mask_boundary_on_both_sides_by_signed_distance(~avoid_set, 1)
    #         & ~
    #         (flag_states_on_grid(
    #             cell_centerpoints=load_certified_states(),
    #             cell_halfwidths=(0.009375, 0.009375, 0.009375, 0.009375),
    #             grid=grid,
    #             verbose=True,
    #             save_array=False
    #         )
    #          & ~
    #          flag_states_on_grid(
    #              cell_centerpoints=load_uncertified_states(),
    #              cell_halfwidths=(0.009375, 0.009375, 0.009375, 0.009375),
    #              grid=grid,
    #              verbose=True,
    #              save_array=False
    #          )
    #          )
    # )
    active_set = jnp.ones_like(avoid_set, dtype=bool)

    # solve
    result = solver(active_set=active_set, initial_values=initial_values)

    if save_result:
        result.save(generate_unique_filename('data/local_update_results/result_qv_sac_global', 'dill'))

    return result


if __name__ == '__main__':
    result_qv_sac_global(save_result=True)
