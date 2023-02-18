import os
import warnings

import hj_reachability
import matplotlib
from jax import numpy as jnp
from matplotlib import pyplot as plt

from refineNCBF.dynamic_systems.implementations.quadcopter_fixed_policy import load_quadcopter_sac_jax_hj
from refineNCBF.refining.hj_reachability_interface.hj_value_postprocessors import ReachAvoid
from refineNCBF.refining.local_hjr_solver.solve import LocalHjrSolver
from refineNCBF.utils.files import generate_unique_filename, visuals_data_directory
from refineNCBF.utils.sets import compute_signed_distance, get_mask_boundary_on_both_sides_by_signed_distance
from refineNCBF.utils.tables import tabularize_dnn, flag_states_on_grid
from refineNCBF.utils.visuals import ArraySlice2D, DimName
from scripts.barrier_refinement.pre_constrcuted_stuff.quadcopter_cbf import load_quadcopter_cbf, load_standardizer, load_uncertified_states, \
    load_certified_states

warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use('TkAgg')


def result_qv_sac_global(save_result: bool = False):
    grid = hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj_reachability.sets.Box(
            [-1, -8, -jnp.pi, -10],
            [11, 8, jnp.pi, 10]
        ),
        shape=(41, 41, 41, 41)
    )

    dynamics = load_quadcopter_sac_jax_hj(grid)

    avoid_set = (
            (grid.states[..., 0] < 0)
            |
            (grid.states[..., 0] > 10)
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
