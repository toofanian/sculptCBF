import os
import warnings

import hj_reachability
from jax import numpy as jnp
import matplotlib
from matplotlib import pyplot as plt

from refineNCBF.dynamic_systems.implementations.active_cruise_control import ActiveCruiseControlJAX, simplified_active_cruise_control_params
from refineNCBF.refining.hj_reachability_interface.hj_dynamics import HJControlAffineDynamics, ActorModes
from refineNCBF.refining.hj_reachability_interface.hj_setup import HjSetup

from refineNCBF.refining.hj_reachability_interface.hj_value_postprocessors import ReachAvoid
from refineNCBF.refining.local_hjr_solver.local_hjr_solver import LocalHjrSolver
from refineNCBF.utils.files import visuals_data_directory, generate_unique_filename
from refineNCBF.utils.sets import compute_signed_distance
from refineNCBF.utils.visuals import ArraySlice2D
from scripts.barrier_refinement.pre_constrcuted_stuff.active_cruise_control_stuff import get_saved_signed_distance_function, SignedDistanceFunctions

warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use("TkAgg")


def demo_local_hjr_classic_solver_on_active_cruise_control(verbose: bool = False, save_gif: bool = False):
    hj_setup = HjSetup.from_parts(
        dynamics=HJControlAffineDynamics.from_control_affine_dynamics(
            control_affine_dynamic_system=ActiveCruiseControlJAX.from_params(simplified_active_cruise_control_params),
            control_mode=ActorModes.MAX,
            disturbance_mode=ActorModes.MIN,
        ),
        grid=hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
            domain=hj_reachability.sets.Box(
                [0, -20, 20],
                [1e3, 20, 80]
            ),
            shape=(21, 21, 21)
        )
    )

    avoid_set = get_saved_signed_distance_function(
        signed_distance_function=SignedDistanceFunctions.X3_DISTANCE_KERNEL_CUT_55dist,
        hj_setup=hj_setup
    ) < 0
    reach_set = jnp.zeros_like(avoid_set, dtype=bool)

    terminal_values = compute_signed_distance(~avoid_set)
    solver_settings = hj_reachability.SolverSettings.with_accuracy(
        accuracy=hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH,
        value_postprocessor=ReachAvoid.from_array(
            values=terminal_values,
            reach_set=reach_set,
        )
    )

    solver = LocalHjrSolver.as_classic_solver(
        hj_setup=hj_setup,
        solver_settings=solver_settings,
        avoid_set=avoid_set,
        reach_set=reach_set,
        max_iterations=20,
        value_change_atol=1e-5,
        value_change_rtol=1e-5,
        verbose=verbose
    )

    initial_values = terminal_values.copy() + 5
    active_set = jnp.ones_like(avoid_set, dtype=bool) & (hj_setup.grid.states[..., 1] < -2.5) & (hj_setup.grid.states[..., 2] > 47.5)

    result = solver(active_set=active_set, initial_values=initial_values)

    if verbose:
        ref_index = ArraySlice2D.from_reference_index(
            reference_index=(3, 0, 0),
            free_dim_1=1,
            free_dim_2=2
        )

        if save_gif:
            result.create_gif(
                reference_slice=ref_index,
                verbose=verbose,
                save_path=os.path.join(
                    visuals_data_directory,
                    f'{generate_unique_filename("demo_local_hjr_solver_classic_on_active_cruise_control", "gif")}'
                )
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

        # result.plot_value_function_against_truth(
        #     reference_slice=ref_index,
        #     verbose=verbose
        # )

        plt.pause(0)

    return result
