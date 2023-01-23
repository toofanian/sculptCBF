import hj_reachability
import jax

from refineNCBF.dynamic_systems.implementations.active_cruise_control import ActiveCruiseControlJAX, simplified_active_cruise_control_params
from refineNCBF.refining.hj_reachability_interface.hj_dynamics import HJControlAffineDynamics, ActorModes
from refineNCBF.refining.hj_reachability_interface.hj_setup import HjSetup
from refineNCBF.refining.hj_reachability_interface.hj_step import hj_step
from refineNCBF.refining.hj_reachability_interface.hj_value_postprocessors import ReachAvoid
from refineNCBF.utils.sets import compute_signed_distance
from scripts.barrier_refinement.pre_constrcuted_stuff.active_cruise_control_stuff import get_saved_signed_distance_function, SignedDistanceFunctions


def test_no_change_where_inactive():
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

    active_set = jax.numpy.ones_like(avoid_set, dtype=bool) & (hj_setup.grid.states[..., 1] < -2.5) & (hj_setup.grid.states[..., 2] > 47.5)

    values_next = hj_step(
        hj_setup=hj_setup,
        solver_settings=solver_settings,
        initial_values=terminal_values.copy(),
        time_start=0,
        time_target=-1,
        active_set=active_set,
        progress_bar=False,
    )

    assert jax.numpy.isclose(values_next[~active_set], terminal_values[~active_set], atol=1e-3).all()
