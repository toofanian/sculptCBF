import hj_reachability
import jax
from refineNCBF.refining.hj_reachability_interface.hj_setup import HjSetup

from refineNCBF.dynamic_systems.implementations.active_cruise_control import ActiveCruiseControlJAX, simplified_active_cruise_control_params
from refineNCBF.refining.hj_reachability_interface.hj_dynamics import HJControlAffineDynamics, ActorModes
from refineNCBF.refining.hj_reachability_interface.hj_value_postprocessors import ReachAvoid
from refineNCBF.refining.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.refining.local_hjr_solver.solve import LocalHjrSolver
from refineNCBF.utils.files import remove_file
from refineNCBF.utils.sets import compute_signed_distance
from scripts.barrier_refinement.pre_constrcuted_stuff.archive_active_cruise_control_stuff import get_saved_signed_distance_function, SignedDistanceFunctions


def test_save_and_load():
    hj_setup = HjSetup.from_parts(
        dynamics=HJControlAffineDynamics.from_parts(
            control_affine_dynamic_system=ActiveCruiseControlJAX.from_params(simplified_active_cruise_control_params),
            control_mode=ActorModes.MAX,
            disturbance_mode=ActorModes.MIN,
        ),
        grid=hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
            domain=hj_reachability.sets.Box(
                [0, -20, 20],
                [1e3, 20, 80]
            ),
            shape=(5, 31, 31)
        )
    )

    avoid_set = get_saved_signed_distance_function(
        signed_distance_function=SignedDistanceFunctions.X3_DISTANCE_KERNEL_CUT_55dist,
        hj_setup=hj_setup
    ) < 0
    reach_set = jax.numpy.zeros_like(avoid_set, dtype=bool)

    terminal_values = compute_signed_distance(~avoid_set)
    solver_settings = hj_reachability.SolverSettings(
        value_postprocessor=ReachAvoid.from_array(
            values=terminal_values,
            reach_set=reach_set,
        )
    )

    solver = LocalHjrSolver.as_local_solver(
        hj_setup=hj_setup,
        solver_settings=solver_settings,
        avoid_set=avoid_set,
        reach_set=reach_set,
        max_iterations=10,
        value_change_atol=1e-5,
        value_change_rtol=1e-5,
        verbose=False
    )

    initial_values = terminal_values.copy() + 5
    active_set = jax.numpy.ones_like(avoid_set, dtype=bool) & (hj_setup.grid.states[..., 1] < -2.5) & (hj_setup.grid.states[..., 2] > 47.5)

    dummy_result = solver(active_set=active_set, initial_values=initial_values)

    path = "tests/test_data/test_local_hjr_result-test_save_and_load-dummy_result.dill"

    dummy_result.save(path)
    loaded_result = LocalUpdateResult.load(path)

    remove_file(path)

    assert jax.numpy.equal(dummy_result.get_recent_values(), loaded_result.get_recent_values()).all()
