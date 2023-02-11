import os
import warnings

import hj_reachability
from jax import numpy as jnp
import matplotlib

from refineNCBF.dynamic_systems.implementations.active_cruise_control_odp import active_cruise_control_odp_dynamics

from refineNCBF.refining.local_hjr_solver.solver_odp import create_global_solver_odp
from refineNCBF.utils.files import visuals_data_directory, generate_unique_filename
from refineNCBF.utils.sets import compute_signed_distance, get_mask_boundary_on_both_sides_by_signed_distance
from refineNCBF.utils.visuals import ArraySlice2D, DimName

warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use("TkAgg")


def demo_local_hjr_classic_solver_on_active_cruise_control(verbose: bool = False, save_gif: bool = False, save_result: bool = False):
    dynamics = active_cruise_control_odp_dynamics

    grid = hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj_reachability.sets.Box(
            [0, -20, 20],
            [1e3, 20, 80]
        ),
        shape=(3, 101, 101)
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
        max_iterations=1,
        verbose=True
    )

    initial_values = terminal_values.copy()
    active_set = get_mask_boundary_on_both_sides_by_signed_distance(~avoid_set, distance=2)

    result = solver(active_set=active_set, initial_values=initial_values)

    if save_result:
        result.save(generate_unique_filename('data/local_update_results/demo_local_hjr_classic_solver_on_active_cruise_control', 'dill'))

    if verbose:
        ref_index = ArraySlice2D.from_reference_index(
            reference_index=(2, 0, 0),
            free_dim_1=DimName(1, 'relative velocity'),
            free_dim_2=DimName(2, 'relative distance'),
        )

        if save_gif:
            result.create_gif(
                reference_slice=ref_index,
                verbose=verbose,
                save_path=os.path.join(
                    visuals_data_directory,
                    f'{generate_unique_filename("acc_odp", "gif")}'
                )
            )
    #     else:
    #         result.create_gif(
    #             reference_slice=ref_index,
    #             verbose=verbose
    #         )
    #
    #     result.plot_value_function(
    #         reference_slice=ref_index,
    #         verbose=verbose
    #     )
    #
    #     result.plot_value_function_against_truth(
    #         reference_slice=ref_index,
    #         levelset=[0],
    #         verbose=verbose
    #     )
    #
    #     result.plot_safe_cells_against_truth(
    #         reference_slice=ref_index,
    #         verbose=verbose
    #     )
    #
    #     plt.pause(0)

    return result


if __name__ == '__main__':
    demo_local_hjr_classic_solver_on_active_cruise_control(verbose=True, save_gif=True, save_result=False)
