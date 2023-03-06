from typing import Tuple, Callable, Optional

import attr
import jax
import numpy as np
from jax import numpy as jnp

import hj_reachability
from refineNCBF.dynamic_systems.dynamic_systems import ControlAffineDynamicSystemFixedPolicy
from refineNCBF.dynamic_systems.quadcopter import QuadcopterVerticalParams, default_quadcopter_vertical_params
from refineNCBF.hj_reachability_interface.hj_dynamics import HJControlAffineDynamicsFixedPolicy, ActorModes
from refineNCBF.utils.files import FilePathRelative
from refineNCBF.utils.types import VectorBatch
from scripts.pre_constructed_stuff.quadcopter_cbf import load_tabularized_sac


@attr.s(auto_attribs=True, eq=False)
class QuadcopterFixedPolicy(ControlAffineDynamicSystemFixedPolicy):
    gravity: float
    mass: float
    drag_coefficient_v: float
    drag_coefficient_phi: float
    length_between_copters: float
    moment_of_inertia: float

    state_dimensions: int = 4
    periodic_state_dimensions: Tuple[float, ...] = (2,)

    control_dimensions: int = 2

    disturbance_dimensions: int = 1
    disturbance_lower_bounds: Tuple[float, ...] = (0,)
    disturbance_upper_bounds: Tuple[float, ...] = (0,)

    _control_policy: Callable[[VectorBatch], VectorBatch] = attr.ib(factory=lambda: lambda x: jnp.zeros((2, 1)))
    _disturbance_policy: Callable[[VectorBatch], VectorBatch] = attr.ib(factory=lambda: lambda x: jnp.zeros((1, 1)))

    @classmethod
    def from_specs_with_policy(
            cls,
            params: QuadcopterVerticalParams,
            control_policy: Optional[Callable[[VectorBatch], VectorBatch]] = None,
            disturbance_policy: Optional[Callable[[VectorBatch], VectorBatch]] = None,
    ) -> 'QuadcopterFixedPolicy':

        if control_policy is None:
            control_policy = lambda x: jnp.zeros((2, 1))
        if disturbance_policy is None:
            disturbance_policy = lambda x: jnp.zeros((1, 1))

        control_lower_bounds = (params.min_thrust, params.min_thrust)
        control_upper_bounds = (params.max_thrust, params.max_thrust)

        return cls(
            gravity=params.gravity,
            mass=params.mass,
            drag_coefficient_v=params.drag_coefficient_v,
            drag_coefficient_phi=params.drag_coefficient_phi,
            length_between_copters=params.length_between_copters,
            moment_of_inertia=params.moment_of_inertia,
            control_lower_bounds=control_lower_bounds,
            control_upper_bounds=control_upper_bounds,
            control_policy=control_policy,
            disturbance_policy=disturbance_policy,
        )

    def compute_open_loop_dynamics(self, state: jax.Array, time: jax.Array = 0.0) -> jax.Array:
        open_loop_dynamics = jnp.array([
            state[1],
            -state[1] * self.drag_coefficient_v / self.mass - self.gravity,
            state[3],
            -state[3] * self.drag_coefficient_phi / self.moment_of_inertia
        ])
        return open_loop_dynamics

    def compute_control_jacobian(self, state, time: jax.Array = 0.0):
        control_jacobian = jnp.array([
            [0, 0],
            [jnp.cos(state[2]) / self.mass, jnp.cos(state[2]) / self.mass],
            [0, 0],
            [-self.length_between_copters / self.moment_of_inertia,
             self.length_between_copters / self.moment_of_inertia]
        ])
        return control_jacobian

    def compute_disturbance_jacobian(self, state: np.ndarray, time: jax.Array = 0.0) -> jax.Array:
        disturbance_jacobian = jnp.expand_dims(jnp.zeros(4), axis=-1)
        return disturbance_jacobian

    def compute_control(self, state: VectorBatch) -> VectorBatch:
        control = jnp.atleast_1d(self._control_policy(state).squeeze())
        return control

    def compute_disturbance(self, state: VectorBatch) -> VectorBatch:
        disturbance = jnp.atleast_1d(self._disturbance_policy(state).squeeze())
        return disturbance


def load_quadcopter_sac_jax_hj(
        grid: hj_reachability.Grid,
        relative_path: FilePathRelative
) -> HJControlAffineDynamicsFixedPolicy:
    return HJControlAffineDynamicsFixedPolicy.from_parts(
        dynamics=QuadcopterFixedPolicy.from_specs_with_policy(
            params=default_quadcopter_vertical_params,
            control_policy=load_tabularized_sac(grid, relative_path),
        ),
        control_mode=ActorModes.MAX,
        disturbance_mode=ActorModes.MIN,
    )
