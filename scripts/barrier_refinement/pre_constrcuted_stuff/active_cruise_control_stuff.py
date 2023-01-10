from enum import IntEnum

import hj_reachability
import numpy as np
import jax.numpy as jnp
import skfmm

from refineNCBF.dynamic_systems.implementations.active_cruise_control import ActiveCruiseControlJAX, \
    simplified_active_cruise_control_params
from refineNCBF.refining.hj_reachability_interface.hj_dynamics import HJControlAffineDynamics, ActorModes
from refineNCBF.refining.hj_reachability_interface.hj_setup import HjSetup
from refineNCBF.refining.hj_reachability_interface.hj_value_postprocessors import NotBiggerator
from refineNCBF.refining.hj_reachability_interface.hj_vanilla_step import hjr_solve_vanilla
from refineNCBF.utils.sets import compute_signed_distance


def acc_set_up_standard_dynamics_and_grid():

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
            shape=(5, 51, 51)
        )
    )


    return hj_setup


class SignedDistanceFunctions(IntEnum):
    X3_DISTANCE = 0
    X3_DISTANCE_KERNEL = 1
    X3_DISTANCE_KERNEL_CUT_50dist = 2  # _distance kernel but cut in half along the _distance axis
    X3_DISTANCE_KERNEL_CUT_55dist = 3  # _distance kernel but cut in half along the _distance axis
    X3_DISTANCE_KERNEL_PLUS = 4
    X3_DISTANCE_5LEVELSET = 5


def get_saved_signed_distance_function(signed_distance_function: SignedDistanceFunctions, hj_setup: HjSetup):


    match signed_distance_function:
        case SignedDistanceFunctions.X3_DISTANCE:
            grid_np = np.array(hj_setup.grid.states)
            where_boundary = np.logical_and(grid_np[:, :, :, 2] > 40, grid_np[:, :, :, 2] < 60)
            signed_distance_to_boundary = -skfmm.distance(~where_boundary) + skfmm.distance(where_boundary)

        case SignedDistanceFunctions.X3_DISTANCE_KERNEL:
            grid_np = np.array(hj_setup.grid.states)
            boundary_for_kernel = np.logical_and(grid_np[:, :, :, 2] > 40, grid_np[:, :, :, 2] < 60)
            signed_distance_to_boundary = compute_signed_distance(boundary_for_kernel)
            solver_settings = hj_reachability.SolverSettings(
                value_postprocessor=NotBiggerator(signed_distance_to_boundary, jnp.ones_like(signed_distance_to_boundary, dtype=bool)),
            )
            where_boundary_values = hjr_solve_vanilla(
                hj_setup=hj_setup,
                solver_settings=solver_settings,
                initial_values=signed_distance_to_boundary,
                time_start=0,
                time_target=-20,
                progress_bar=False
            ) > 0

            where_boundary = where_boundary_values
            signed_distance_to_boundary = compute_signed_distance(where_boundary)

        case SignedDistanceFunctions.X3_DISTANCE_KERNEL_CUT_50dist:
            grid_np = np.array(hj_setup.grid.states)
            boundary_for_kernel = np.logical_and(grid_np[:, :, :, 2] > 40, grid_np[:, :, :, 2] < 60)
            signed_distance_to_boundary = compute_signed_distance(boundary_for_kernel)
            solver_settings = hj_reachability.SolverSettings(
                value_postprocessor=NotBiggerator(signed_distance_to_boundary, jnp.ones_like(signed_distance_to_boundary, dtype=bool)),
            )
            where_boundary_values = hjr_solve_vanilla(
                hj_setup=hj_setup,
                solver_settings=solver_settings,
                initial_values=signed_distance_to_boundary,
                time_start=0,
                time_target=-20,
                progress_bar=False
            ) > 0
            where_boundary_states = (grid_np[:, :, :, 2] < 50)
            where_boundary = np.logical_and(where_boundary_states, where_boundary_values)
            signed_distance_to_boundary = compute_signed_distance(where_boundary)

        case SignedDistanceFunctions.X3_DISTANCE_KERNEL_CUT_55dist:
            grid_np = np.array(hj_setup.grid.states)
            boundary_for_kernel = np.logical_and(grid_np[:, :, :, 2] > 40, grid_np[:, :, :, 2] < 60)
            signed_distance_to_boundary = compute_signed_distance(boundary_for_kernel)
            solver_settings = hj_reachability.SolverSettings(
                value_postprocessor=NotBiggerator(signed_distance_to_boundary, jnp.ones_like(signed_distance_to_boundary, dtype=bool)),
            )
            where_boundary_values = hjr_solve_vanilla(
                hj_setup=hj_setup,
                solver_settings=solver_settings,
                initial_values=signed_distance_to_boundary,
                time_start=0,
                time_target=-20,
                progress_bar=False
            ) > 0
            where_boundary_states = (grid_np[:, :, :, 2] < 55)
            where_boundary = np.logical_and(where_boundary_states, where_boundary_values)
            signed_distance_to_boundary = compute_signed_distance(where_boundary)

        case _:
            raise ValueError(f'Unknown signed distance function {signed_distance_function}')

    return jnp.array(signed_distance_to_boundary)