import os
import warnings

import hj_reachability

from refineNCBF.dynamic_systems.implementations.quadcopter import quadcopter_vertical_jax_hj
from refineNCBF.refining.hj_reachability_interface.hj_setup import HjSetup

import jax.numpy as jnp

from refineNCBF.refining.hj_reachability_interface.hj_value_postprocessors import ReachAvoid
from refineNCBF.refining.local_hjr_solver.local_hjr_solver import LocalHjrSolver
from refineNCBF.utils.files import visuals_data_directory, generate_unique_filename
from refineNCBF.utils.sets import compute_signed_distance
from refineNCBF.utils.visuals import ArraySlice2D
from scripts.barrier_refinement.pre_constrcuted_stuff.quadcopter_vertical_stuff import tabularize_vector_to_scalar_mapping, quadcopter_cbf_from_refine_cbf

warnings.simplefilter(action='ignore', category=FutureWarning)


def demo_local_hjr_boundary_decrease_solver_quadcopter_vertical(verbose: bool = False, save_gif: bool = False):
    # set up dynamics and grid
    hj_setup = HjSetup.from_parts(
        dynamics=quadcopter_vertical_jax_hj,
        grid=hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
            domain=hj_reachability.sets.Box(
                [0, -8, -jnp.pi, -10],
                [10, 8, jnp.pi, 10]
            ),
            shape=(21, 21, 21, 21)
        )
    )

    # define reach and avoid targets
    avoid_set = (
            (hj_setup.grid.states[..., 0] < 1)
            |
            (hj_setup.grid.states[..., 0] > 9)
    )
    reach_set = jnp.zeros_like(avoid_set, dtype=bool)

    # create solver settings for backwards reachable tube
    terminal_values = compute_signed_distance(~avoid_set)
    solver_settings = hj_reachability.SolverSettings(
        value_postprocessor=ReachAvoid.from_array(
            values=terminal_values,
            reach_set=reach_set,
        )
    )

    # load into solver
    solver = LocalHjrSolver.as_boundary_solver_with_only_decrease(
        hj_setup=hj_setup,
        solver_settings=solver_settings,
        avoid_set=avoid_set,
        reach_set=reach_set,
        verbose=verbose
    )

    # define initial values and initial active set to solve on
    initial_values = tabularize_vector_to_scalar_mapping(
        mapping=quadcopter_cbf_from_refine_cbf,
        grid=hj_setup.grid
    )
    active_set = jnp.ones_like(hj_setup.grid.states[..., 0], dtype=bool)

    # solve
    result = solver(active_set=active_set, initial_values=initial_values)

    # visualize
    if verbose:
        ref_index = ArraySlice2D.from_reference_index(
            reference_index=(
                jnp.array(hj_setup.grid.states.shape[0]) // 2,
                jnp.array(hj_setup.grid.states.shape[1]) // 4,
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
                    f'{generate_unique_filename("demo_local_hjr_solver_quadcopter_vertical_classic", "gif")}')
            )
        else:
            result.create_gif(
                reference_slice=ref_index,
                verbose=verbose
            )

        result.plot_value_function_against_truth(
            reference_slice=ref_index,
            verbose=verbose
        )

    return result


if __name__ == '__main__':
    demo_local_hjr_boundary_decrease_solver_quadcopter_vertical(verbose=True, save_gif=True)
