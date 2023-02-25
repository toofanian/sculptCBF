import os
import warnings

import jax.numpy as jnp
import matplotlib
from matplotlib import pyplot as plt

import hj_reachability
from refineNCBF.dynamic_systems.go_left import go_left_jax_hj
from refineNCBF.hj_reachability_interface.hj_value_postprocessors import ReachAvoid
from refineNCBF.local_hjr_solver.solve import LocalHjrSolver
from refineNCBF.utils.files import visuals_data_directory, generate_unique_filename
from refineNCBF.utils.sets import compute_signed_distance
from refineNCBF.utils.visuals import ArraySlice1D, DimName

warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use('TkAgg')


def demo_local_hjr_boundary_decrease_solver_go_left(verbose: bool = False, save_gif: bool = False, save_result: bool = False):
    # set up dynamics and grid
    dynamics = go_left_jax_hj

    grid = hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj_reachability.sets.Box(
            [-10],
            [100]
        ),
        shape=(110,)
    )

    # define reach and avoid targets
    avoid_set = (
        (grid.states[..., 0] < 0)
    )
    reach_set = jnp.zeros_like(avoid_set, dtype=bool)

    # create solver settings for backwards reachable tube
    terminal_values = compute_signed_distance(~avoid_set)
    solver_settings = hj_reachability.SolverSettings.with_accuracy(
        hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH,
        value_postprocessor=ReachAvoid.from_array(
            values=terminal_values,
            reach_set=reach_set,
        ),
    )

    # load into solver
    solver = LocalHjrSolver.as_boundary_decrease(
        dynamics=dynamics,
        grid=grid,
        solver_settings=solver_settings,
        avoid_set=avoid_set,
        reach_set=reach_set,
        verbose=verbose,
        solver_timestep=-1,
        neighbor_distance=5,
        boundary_distance=5,
        max_iterations=100
    )

    # define initial values and initial active set to solve on
    initial_values = terminal_values.copy()
    active_set = jnp.zeros_like(grid.states[..., 0], dtype=bool).at[10].set(True)

    # solve
    result = solver(active_set=active_set, initial_values=initial_values)

    if save_result:
        result.save(generate_unique_filename('data/local_update_results/demo_local_hjr_boundary_decrease_solver_go_left', 'dill'))

    # visualize
    if verbose:
        ref_index = ArraySlice1D.from_reference_index(
            reference_index=(10,),
            free_dim_1=DimName(0, 'x'),
        )

        if save_gif:
            result.create_gif(
                reference_slice=ref_index,
                verbose=verbose,
                save_path=os.path.join(
                    visuals_data_directory,
                    f'{generate_unique_filename("demo_local_hjr_boundary_decrease_solver_go_left", "gif")}')
            )
        else:
            result.create_gif(
                reference_slice=ref_index,
                verbose=verbose
            )

        result.plot_value_1d(
            ref_index
        )

    plt.pause(0)

    return result


if __name__ == '__main__':
    demo_local_hjr_boundary_decrease_solver_go_left(verbose=True, save_gif=True, save_result=True)
