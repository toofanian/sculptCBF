import warnings

import hj_reachability
import jax.numpy as jnp

# from verify_ncbf.toof.scripts.utils.acc_hjr_utils import acc_set_up_standard_dynamics_and_grid
# from verify_ncbf.toof.scripts.utils.acc_signed_distance_functions import get_saved_signed_distance_function, SignedDistanceFunctions
# from verify_ncbf.toof.scripts.utils.quadcopter_hjr_utils import make_hj_setup_quadcopter_vertical, tabularize_vector_to_scalar_mapping, \
#     quadcopter_cbf_from_refine_cbf
# from verify_ncbf.toof.src.dynamic_systems.implementations.quadcopter import quadcopter_vertical_jax_hj
# from verify_ncbf.toof.src.refining.oop_refactor_for_local_update.active_set_post_filter import RemoveWhereUnchanged, RemoveWhereUnchangedOrOscillating
# from verify_ncbf.toof.src.refining.oop_refactor_for_local_update.active_set_pre_filter import FilterWhereFarFromZeroLevelset, NoFilter
# from verify_ncbf.toof.src.refining.oop_refactor_for_local_update.break_criteria_checker import BreakCriteriaChecker, MaxIterations, \
#     PostFilteredActiveSetEmpty
# from verify_ncbf.toof.src.refining.oop_refactor_for_local_update.local_hjr_solver import LocalHjrSolver
# from verify_ncbf.toof.src.refining.oop_refactor_for_local_update.local_hjr_stepper import ClassicLocalHjrStepper, OnlyDecreaseLocalHjrStepper
# from verify_ncbf.toof.src.refining.oop_refactor_for_local_update.neighbor_expander import SignedDistanceNeighbors
# from verify_ncbf.toof.src.refining.reachability_utils.hj_setup import HjSetup
# from verify_ncbf.toof.src.refining.reachability_utils.value_postprocessors import ReachAvoid
# from verify_ncbf.toof.src.utils.drawing.drawing_utils import ArraySlice2D
# from verify_ncbf.toof.src.utils.files.file_path import generate_unique_filename
# from verify_ncbf.toof.src.utils.math.signed_distance import compute_signed_distance




warnings.simplefilter(action='ignore', category=FutureWarning)

"""
local hjr solver types:
    * CLASSIC
        emulates the algorithm from somil & bajcsy
        
        algorithm:
            - expand active set to neighbors
            - compute vanilla hjr value update over active set
            - remove unchanged values from active set
            - repeat until active set is empty

    * ONLY DECREASE
        emulates the algorithm from somil & bajcsy, but with modified value update which only 
        accepts decreasing values. has the effect of only updating near the running zero 
        levelset, since the zero levelset drives the values to decrease. the values far from
        the viability kernel may be invalid because of this.
        
        notable side-effect: the viability kernel will be conservative (too small) 
        if the initial zero levelset is not a complete superset of the viability kernel.
        this is ok when finding the maximal control invariant subset of an initial set.
        
        notable perk: forcing the values to only decrease gets rid of the local-oscillation
        artifacts that occur in the classic algorithm, which prevented convergence detection.
        
        algorithm:
            - expand active set to neighbors
            - compute the only-decrease hjr value update over active set
            - remove unchanged values from active set
            - repeat until active set is empty
            
    * ONLY ACTIVE NEAR ZERO LEVELSET
        emulates the algorithm from somil & bajcsy, but with added active set filtering. the 
        active set is filtered before each iteration to only include values near the zero
        levelset. this is a like a hacky/forced version of only decrease, but has the benefit
        of not requiring the initial zero levelset to be a complete superset of the viability
        kernel. any guess will eventually lead to the correct viability kernel. the values far 
        from the viability kernel may be invalid because of this.
        
        algorithm:
            - pre-filter active set to only include values near the zero levelset
            - expand active set to neighbors
            - compute vanilla hjr value update over active set
            - remove unchanged values from active set
            - repeat until active set is empty
            
"""


def demo_local_hjr_solver_quadcopter_vertical_classic(verbose: bool = False, save_gif: bool = False):
    """
    Demo the classic local HJR solver on the quadcopter vertical problem.
    :param verbose:
    :param save_gif:
    :return:
    """

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

    avoid_set = (
            (hj_setup.grid.states[..., 0] < 1)
            |
            (hj_setup.grid.states[..., 0] > 9)
    )
    reach_set = jnp.zeros_like(avoid_set, dtype=bool)

    terminal_values = compute_signed_distance(~avoid_set)
    solver_settings = hj_reachability.SolverSettings(
        value_postprocessor=ReachAvoid.from_array(
            values=terminal_values,
            reach_set=reach_set,
        )
    )

    solver = LocalHjrSolver.from_parts(
        hj_setup=hj_setup,
        solver_settings=solver_settings,
        avoid_set=avoid_set,
        reach_set=reach_set,

        active_set_pre_filter=NoFilter.from_parts(),
        neighbor_expander=SignedDistanceNeighbors.from_parts(distance=1.0),
        local_hjr_stepper=OnlyDecreaseLocalHjrStepper.from_parts(
            hj_setup=hj_setup,
            solver_settings=solver_settings,
            time_step=-0.1,
            verbose=verbose
        ),
        active_set_post_filter=RemoveWhereUnchanged.from_parts(atol=1e-3, rtol=1e-3),
        break_criteria_checker=BreakCriteriaChecker.from_criteria(
            [
                MaxIterations.from_parts(max_iterations=100),
                PostFilteredActiveSetEmpty.from_parts()
            ]
        )
    )

    initial_values = tabularize_vector_to_scalar_mapping(
        mapping=quadcopter_cbf_from_refine_cbf,
        grid=hj_setup.grid
    )
    active_set = jnp.ones_like(hj_setup.grid.states, dtype=bool)

    result = solver(active_set=active_set, initial_values=initial_values)

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

        result.create_gif(
            reference_slice=ref_index,
            verbose=verbose,
            save_path=f'verify_ncbf/toof/data/{generate_unique_filename("demo_quadcopter_vertical_avoid_hjr_local", "gif")}' if save_gif else None

        )

        result.plot_value_function_against_truth(
            reference_slice=ref_index,
            verbose=verbose
        )

    return result


def demo_local_hjr_solver_custom_on_quadcopter_vertical(verbose: bool = False, save_gif: bool = False):
    hj_setup = make_hj_setup_quadcopter_vertical()

    avoid_set = (
            (hj_setup.grid.states[..., 0] < 1)
            |
            (hj_setup.grid.states[..., 0] > 9)
    )

    terminal_values = compute_signed_distance(~avoid_set)

    reach_set = jnp.zeros_like(avoid_set, dtype=bool)
    active_set = jnp.ones_like(avoid_set, dtype=bool)
    solver_settings = hj_reachability.SolverSettings(
        value_postprocessor=ReachAvoid.from_array(
            values=terminal_values,
            reach_set=reach_set,
        )
    )

    solver = LocalHjrSolver.from_parts(
        hj_setup=hj_setup,
        solver_settings=solver_settings,
        avoid_set=avoid_set,
        reach_set=reach_set,

        active_set_pre_filter=FilterWhereFarFromZeroLevelset.from_parts(distance=1.0),
        neighbor_expander=SignedDistanceNeighbors.from_parts(distance=1.0),
        local_hjr_stepper=ClassicLocalHjrStepper.from_parts(
            hj_setup=hj_setup,
            solver_settings=solver_settings,
            time_step=-0.1,
            verbose=verbose
        ),
        active_set_post_filter=RemoveWhereUnchangedOrOscillating.from_parts(atol=1e-3, rtol=1e-3, history_length=10,
                                                                            std_threshold=1e-2),
        break_criteria_checker=BreakCriteriaChecker.from_criteria(
            [
                MaxIterations.from_parts(max_iterations=50),
                PostFilteredActiveSetEmpty.from_parts()
            ]
        )
    )

    result = solver(active_set=active_set, initial_values=terminal_values)

    if verbose:
        middle_indices = jnp.array(hj_setup.grid.states.shape) // 2
        ref_index = ArraySlice2D.from_reference_index(
            reference_index=(middle_indices[0], middle_indices[1], middle_indices[2], middle_indices[3]),
            free_dim_1=0,
            free_dim_2=2
        )

        result.create_gif(
            reference_slice=ref_index,
            verbose=verbose,
            save_path=f'verify_ncbf/toof/data/{generate_unique_filename("demo_quadcopter_vertical_avoid_hjr_local", "gif")}' if save_gif else None

        )

        result.plot_value_function_against_truth(
            reference_slice=ref_index,
            verbose=verbose
        )

    return result


def demo_local_hjr_solver_custom_on_active_cruise_control(verbose: bool = False):
    hj_setup = acc_set_up_standard_dynamics_and_grid()

    terminal_values = get_saved_signed_distance_function(
        signed_distance_function=SignedDistanceFunctions.X3_DISTANCE
    )

    avoid_set = terminal_values < 0
    reach_set = jnp.zeros_like(avoid_set, dtype=bool)
    active_set = jnp.ones_like(avoid_set, dtype=bool)
    solver_settings = hj_reachability.SolverSettings(
        value_postprocessor=ReachAvoid.from_array(
            values=terminal_values,
            reach_set=reach_set,
        )
    )

    solver = LocalHjrSolver.from_parts(
        hj_setup=hj_setup,
        solver_settings=solver_settings,
        avoid_set=avoid_set,
        reach_set=reach_set,

        active_set_pre_filter=FilterWhereFarFromZeroLevelset.from_parts(distance=1.0),
        neighbor_expander=SignedDistanceNeighbors.from_parts(distance=2.0),
        local_hjr_stepper=ClassicLocalHjrStepper.from_parts(
            hj_setup=hj_setup,
            solver_settings=solver_settings,
            time_step=-0.1,
            verbose=verbose
        ),
        active_set_post_filter=RemoveWhereUnchanged.from_parts(atol=1e-3, rtol=1e-3),
        break_criteria_checker=BreakCriteriaChecker.from_criteria(
            [
                MaxIterations.from_parts(max_iterations=100),
                PostFilteredActiveSetEmpty.from_parts()
            ]
        )
    )

    result = solver(active_set=active_set, initial_values=terminal_values)

    if verbose:
        ref_index = ArraySlice2D.from_reference_index(
            reference_index=(3, 0, 0),
            free_dim_1=1,
            free_dim_2=2
        )

        result.create_gif(
            reference_slice=ref_index,
            verbose=verbose
        )

        result.plot_value_function_against_truth(
            reference_slice=ref_index,
            verbose=verbose
        )

    return result


def demo_local_hjr_solver_only_decrease_on_active_cruise_control(verbose: bool = False, save_gif: bool = False):
    hj_setup = acc_set_up_standard_dynamics_and_grid()

    terminal_values = get_saved_signed_distance_function(
        signed_distance_function=SignedDistanceFunctions.X3_DISTANCE
    )

    # avoid_set = terminal_values < 0
    # reach_set = jnp.zeros_like(avoid_set, dtype=bool)
    # active_set = jnp.ones_like(avoid_set, dtype=bool)

    avoid_set = terminal_values < 0
    reach_set = (hj_setup.grid.states[..., 2] > 57) & (hj_setup.grid.states[..., 2] < 60) & ~avoid_set
    active_set = jnp.ones_like(~avoid_set, dtype=bool)

    solver_settings = hj_reachability.SolverSettings(
        value_postprocessor=ReachAvoid.from_array(
            values=terminal_values,
            reach_set=reach_set,
        )
    )

    solver = LocalHjrSolver.as_boundary_solver_with_only_decrease(
        hj_setup=hj_setup,
        solver_settings=solver_settings,
        avoid_set=avoid_set,
        reach_set=reach_set,
    )

    result = solver(active_set=active_set, initial_values=terminal_values)

    if verbose:
        ref_index = ArraySlice2D.from_reference_index(
            reference_index=(3, 0, 0),
            free_dim_1=1,
            free_dim_2=2
        )

        result.create_gif(
            reference_slice=ref_index,
            verbose=verbose
        )

        result.plot_value_function_against_truth(
            reference_slice=ref_index,
            verbose=verbose
        )

    return result


def demo_local_hjr_solver_classic_on_active_cruise_control(verbose: bool = False, save_gif: bool = False):
    hj_setup = acc_set_up_standard_dynamics_and_grid()

    terminal_values = get_saved_signed_distance_function(
        signed_distance_function=SignedDistanceFunctions.X3_DISTANCE_KERNEL_CUT_55dist
    )

    avoid_set = terminal_values < 0
    reach_set = jnp.zeros_like(avoid_set, dtype=bool)
    active_set = jnp.ones_like(avoid_set, dtype=bool)
    solver_settings = hj_reachability.SolverSettings(
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
    )

    result = solver(active_set=active_set, initial_values=terminal_values)

    if verbose:
        ref_index = ArraySlice2D.from_reference_index(
            reference_index=(3, 0, 0),
            free_dim_1=1,
            free_dim_2=2
        )

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
    demo_local_hjr_solver_custom_on_active_cruise_control(verbose=True)
    # demo_local_hjr_solver_classic_on_active_cruise_control(verbose=True)
    # demo_local_hjr_solver_custom_on_quadcopter_vertical(verbose=True, save_gif=True)
    # demo_local_hjr_solver_only_decrease_on_active_cruise_control(verbose=True, save_gif=True)
