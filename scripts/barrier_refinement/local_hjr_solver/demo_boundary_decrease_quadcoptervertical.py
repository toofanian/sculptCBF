import os
import warnings

import hj_reachability
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from refineNCBF.dynamic_systems.implementations.quadcopter import quadcopter_vertical_jax_hj

import jax.numpy as jnp

from refineNCBF.refining.hj_reachability_interface.hj_step import hj_step
from refineNCBF.refining.hj_reachability_interface.hj_value_postprocessors import ReachAvoid
from refineNCBF.refining.local_hjr_solver.breaker import BreakCriteriaChecker, MaxIterations, PostFilteredActiveSetEmpty
from refineNCBF.refining.local_hjr_solver.expand import SignedDistanceNeighborsNearBoundary
from refineNCBF.refining.local_hjr_solver.postfilter import RemoveWhereNonNegativeHamiltonian
from refineNCBF.refining.local_hjr_solver.prefilter import NoPreFilter, PreFilterWhereFarFromBoundarySplit
from refineNCBF.refining.local_hjr_solver.solve import LocalHjrSolver
from refineNCBF.refining.local_hjr_solver.step import DecreaseReplaceLocalHjrStepper, DecreaseLocalHjrStepper
from refineNCBF.utils.files import visuals_data_directory, generate_unique_filename, construct_full_path
from refineNCBF.utils.sets import compute_signed_distance, get_mask_boundary_on_both_sides_by_signed_distance, expand_mask_by_signed_distance
from refineNCBF.utils.visuals import ArraySlice2D, DimName

warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use('TkAgg')


def get_warmstart_boundary_for_quadcopter(dynamics, grid):
    avoid_set = (
            (grid.states[..., 0] < 1)
            |
            (grid.states[..., 0] > 9)
    )
    terminal_values = compute_signed_distance(~avoid_set)

    # solver_settings = hj_reachability.solver.SolverSettings.with_accuracy(
    #     accuracy=hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH,
    #     value_postprocessor=ReachAvoid.from_array(terminal_values)
    # )
    # truth = hj_step(
    #     dynamics=dynamics,
    #     grid=grid,
    #     solver_settings=solver_settings,
    #     initial_values=terminal_values,
    #     time_start=0.,
    #     time_target=-10,
    #     progress_bar=True
    # )

    truth = jnp.array(np.load(construct_full_path("data/visuals/truth_20230210_124946.npy")))

    kernel = truth >= 0

    oops_im_adding_leaky_stuff = (
        (grid.states[..., 2] < 2)
        &
        (grid.states[..., 2] > 1)
        &
        (grid.states[..., 0] > 2)
        &
        (grid.states[..., 0] < 9)
    )

    leaky_kernel = kernel | oops_im_adding_leaky_stuff
    signed_distance = compute_signed_distance(leaky_kernel)

    where_added = leaky_kernel & ~kernel
    new_terminal_values = truth.at[where_added].set(signed_distance[where_added])

    active_set = get_mask_boundary_on_both_sides_by_signed_distance(leaky_kernel, distance=2) & expand_mask_by_signed_distance(where_added, 2)

    return new_terminal_values, (new_terminal_values < 0), active_set


def demo_local_hjr_boundary_decrease_solver_quadcopter_vertical(verbose: bool = False, save_gif: bool = False, save_result: bool = False):
    # set up dynamics and grid
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

    # create solver settings for backwards reachable tube
    # terminal_values, avoid_set, active_set = get_warmstart_boundary_for_quadcopter(dynamics, grid)
    terminal_values = compute_signed_distance(~avoid_set)
    initial_values = terminal_values.copy()
    active_set = get_mask_boundary_on_both_sides_by_signed_distance(~avoid_set, distance=2)


    solver = LocalHjrSolver.from_parts(
        dynamics=dynamics,
        grid=grid,
        avoid_set=avoid_set,
        reach_set=reach_set,
        terminal_values=terminal_values,

        prefilter=PreFilterWhereFarFromBoundarySplit.from_parts(
            distance_inner=2,
            distance_outer=2
        ),
        expander=SignedDistanceNeighborsNearBoundary.from_parts(
            neighbor_distance=2,
            boundary_distance_inner=2,
            boundary_distance_outer=2,
        ),
        stepper=DecreaseLocalHjrStepper.from_parts(
            dynamics=dynamics,
            grid=grid,
            terminal_values=terminal_values,
            time_step=-.1,
            verbose=verbose,
        ),
        postfilter=RemoveWhereNonNegativeHamiltonian.from_parts(
            hamiltonian_atol=1e-3,
        ),
        breaker=BreakCriteriaChecker.from_criteria(
            [
                MaxIterations.from_parts(max_iterations=200),
                PostFilteredActiveSetEmpty.from_parts(),
            ],
            verbose=verbose
        ),

        verbose=verbose,
    )

    # solve
    result = solver(active_set=active_set, initial_values=initial_values)

    if save_result:
        result.save(generate_unique_filename('data/local_update_results/demo_local_hjr_boundary_decrease_solver_quadcopter_vertical', 'dill'))

    # visualize
    if verbose:
        ref_index = ArraySlice2D.from_reference_index(
            reference_index=(15, 12, 20, 12),
            free_dim_1=DimName(0, 'y'),
            free_dim_2=DimName(2, 'theta')
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

        result.plot_value_function(
            reference_slice=ref_index,
            verbose=verbose
        )

        result.plot_safe_cells_against_truth(
            reference_slice=ref_index,
            verbose=verbose
        )

    plt.pause(0)
    return result


if __name__ == '__main__':
    demo_local_hjr_boundary_decrease_solver_quadcopter_vertical(verbose=True, save_gif=True, save_result=True)
