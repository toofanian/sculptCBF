import functools
from typing import Optional

import hj_reachability
import jax
import numpy as np
from hj_reachability import utils
from hj_reachability.solver import _try_get_progress_bar, nullcontext, TqdmWrapper
from jax import numpy as jnp

from refineNCBF.refining.hj_reachability_interface.hj_setup import HjSetup
from refineNCBF.utils.types import ArrayNd, MaskNd


@functools.partial(jax.jit, static_argnames=("hj_setup", "progress_bar"))
def hjr_step_local(
        solver_settings: hj_reachability.SolverSettings,
        hj_setup: HjSetup,
        start_time: float,
        values: ArrayNd,
        target_time: float,
        active_set: MaskNd,
        progress_bar=True
) -> ArrayNd:
    with (_try_get_progress_bar(start_time, target_time) if progress_bar is True else nullcontext(progress_bar)) as bar:
        def sub_step(time_values):
            time_initial = time_values[0]
            values_initial = time_values[1]

            time_substepped, values_substepped = third_order_total_variation_diminishing_runge_kutta_local(
                solver_settings=solver_settings,
                dynamics=hj_setup.dynamics,
                grid=hj_setup.grid,
                initial_time=time_initial,
                initial_values=values_initial,
                target_time=target_time,
                active_set=active_set,
            )

            if isinstance(bar, TqdmWrapper):
                bar.update_to(jnp.abs(time_substepped - bar.reference_time))

            return time_substepped, values_substepped

        final_time, final_values = jax.lax.while_loop(
            cond_fun=lambda time_values_tuple: jnp.abs(target_time - time_values_tuple[0]) > 0,
            body_fun=sub_step,
            init_val=(start_time, values)
        )

        return final_values


def third_order_total_variation_diminishing_runge_kutta_local(
        solver_settings: hj_reachability.SolverSettings,
        dynamics: hj_reachability.Dynamics,
        grid: hj_reachability.Grid,
        initial_time: float,
        initial_values: ArrayNd,
        target_time: float,
        active_set: MaskNd
):
    time_1, values_1 = euler_step_local(solver_settings, dynamics, grid, initial_time, initial_values,
                                        max_time_step=target_time - initial_time, active_set=active_set)
    time_step = time_1 - initial_time
    _, values_2 = euler_step_local(solver_settings, dynamics, grid, time_1, values_1, time_step, active_set=active_set)
    time_0_5, values_0_5 = initial_time + time_step / 2, (3 / 4) * initial_values + (1 / 4) * values_2
    _, values_1_5 = euler_step_local(solver_settings, dynamics, grid, time_0_5, values_0_5, time_step,
                                     active_set=active_set)
    return time_1, solver_settings.value_postprocessor(time_1, (1 / 3) * initial_values + (2 / 3) * values_1_5)


@functools.partial(jax.jit, static_argnames="dynamics")
def euler_step_local(
        solver_settings: hj_reachability.SolverSettings,
        dynamics: hj_reachability.Dynamics,
        grid: hj_reachability.Grid,
        time: float,
        values: ArrayNd,
        time_step: float = None,
        max_time_step: float = None,
        active_set: Optional[MaskNd] = None
):
    time_direction = jnp.sign(max_time_step) if time_step is None else jnp.sign(time_step)

    def signed_lax_friedrichs_numerical_hamiltonian(
            _state, _time, _value, _left_grad_value, _right_grad_value, _dissipation_coefficients
    ):
        hamiltonian_value = time_direction * dynamics.hamiltonian(_state, _time, _value,
                                                                  (_left_grad_value + _right_grad_value) / 2)
        dissipation_value = _dissipation_coefficients @ (_right_grad_value - _left_grad_value) / 2
        return hamiltonian_value - dissipation_value

    left_grad_values, right_grad_values = grid.upwind_grad_values(
        solver_settings.upwind_scheme, values
    )

    dissipation_coefficients = solver_settings.artificial_dissipation_scheme(
        dynamics.partial_max_magnitudes, grid.states, time, values, left_grad_values, right_grad_values
    )

    # TODO: this next bit doesnt seem to speed up the compute, but it should...
    #       maybe an issue with some cores taking longer than others, holding up each batch

    def local_hammy(
            _state, _value, _left_grad_value, _right_grad_value, _dissipation_coefficients, _active_status
    ):
        hammy = jax.lax.cond(
            _active_status,
            signed_lax_friedrichs_numerical_hamiltonian,
            lambda *args, **kwargs: jnp.array(0, dtype=jnp.float32),
            *[_state, time, _value, _left_grad_value, _right_grad_value, _dissipation_coefficients]
        )
        return hammy

    dvalues_dt = time_direction * utils.multivmap(
        fun=local_hammy,
        in_axes=np.arange(grid.ndim)
    )(
        grid.states,
        values,
        left_grad_values,
        right_grad_values,
        dissipation_coefficients,
        active_set
    )

    dvalues_dt = -solver_settings.hamiltonian_postprocessor(dvalues_dt)

    if time_step is None:
        time_step_bound = 1 / jnp.max(jnp.sum(dissipation_coefficients / jnp.array(grid.spacings), -1))
        time_step = time_direction * jnp.minimum(solver_settings.CFL_number * time_step_bound, jnp.abs(max_time_step))

    return time + time_step, values + time_step * dvalues_dt
