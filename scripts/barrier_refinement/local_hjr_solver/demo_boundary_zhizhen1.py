import os
import warnings

import hj_reachability
import matplotlib
from jax import numpy as jnp
from matplotlib import pyplot as plt

from refineNCBF.dynamic_systems.implementations.quadcopter import quadcopter_vertical_jax_hj
from refineNCBF.refining.hj_reachability_interface.hj_setup import HjSetup
from refineNCBF.refining.hj_reachability_interface.hj_value_postprocessors import ReachAvoid
from refineNCBF.refining.local_hjr_solver.local_hjr_solver import LocalHjrSolver
from refineNCBF.utils.files import generate_unique_filename, visuals_data_directory
from refineNCBF.utils.sets import compute_signed_distance
from refineNCBF.utils.tables import tabularize_dnn, flag_states_on_grid
from refineNCBF.utils.visuals import ArraySlice2D
from scripts.barrier_refinement.pre_constrcuted_stuff.quadcopter_cbf import load_quadcopter_cbf, load_standardizer, load_uncertified_states

warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use('TkAgg')


def demo_local_hjr_decrease_solver_on_quadcopter_vertical_ncbf(verbose: bool = False, save_gif: bool = False, save_result: bool = False):
    # set up dynamics and grid
    hj_setup = HjSetup.from_parts(
        dynamics=quadcopter_vertical_jax_hj,
        grid=hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
            domain=hj_reachability.sets.Box(
                [4, -3.3, -1.5, -1],
                [10.5, 2.5, 1.5, 3]
            ),
            shape=(25, 25, 25, 25)  # (31, 25, 41, 25)
        )
    )

    dnn_values_over_grid = -tabularize_dnn(
        dnn=load_quadcopter_cbf(),
        standardizer=load_standardizer(),
        grid=hj_setup.grid
    )

    # define reach and avoid targets
    avoid_set = (dnn_values_over_grid < 0)
    reach_set = jnp.zeros_like(avoid_set, dtype=bool)

    # create solver settings for backwards reachable tube
    terminal_values = compute_signed_distance(~avoid_set)
    solver_settings = hj_reachability.SolverSettings.with_accuracy(
        accuracy=hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH,
        value_postprocessor=ReachAvoid.from_array(
            values=terminal_values,
            reach_set=reach_set,
        )
    )

    # load into solver
    solver = LocalHjrSolver.as_only_decrease(
        hj_setup=hj_setup,
        solver_settings=solver_settings,
        avoid_set=avoid_set,
        reach_set=reach_set,
        verbose=verbose,
        max_iterations=1000,
    )

    # define initial values and initial active set to solve on
    initial_values = terminal_values.copy()
    active_set = flag_states_on_grid(
        cell_centerpoints=load_uncertified_states(),
        cell_halfwidths=(0.009375, 0.009375, 0.009375, 0.009375),
        grid=hj_setup.grid,
        verbose=True,
        save_array=False
    )

    # solve
    result = solver(active_set=active_set, initial_values=initial_values)

    if save_result:
        result.save(generate_unique_filename('data/local_update_results/demo_local_hjr_boundary_decrease_solver_on_quadcopter_vertical_ncbf', 'dill'))

    # visualize
    if verbose:
        ref_index = ArraySlice2D.from_reference_index(
            reference_index=(
                jnp.array(hj_setup.grid.states.shape[0]) // 2,
                jnp.array(9),
                jnp.array(hj_setup.grid.states.shape[2]) // 2,
                jnp.array(hj_setup.grid.states.shape[3]) // 2,
            ),
            free_dim_1=0,
            free_dim_2=2
        )

        if save_gif:
            result.create_gif(
                reference_slice=ref_index,
                verbose=verbose,
                save_path=os.path.join(
                    visuals_data_directory,
                    f'{generate_unique_filename("demo_local_hjr_boundary_decrease_solver_on_quadcopter_vertical_ncbf", "gif")}')
            )
        else:
            result.create_gif(
                reference_slice=ref_index,
                verbose=verbose
            )

        plt.pause(0)

    return result
