from typing import Tuple

import attr
import jax
import jax.numpy as jnp
import numpy as np

from cbf_opt import ControlAffineDynamics
from refineNCBF.hj_reachability_interface.hj_dynamics import HJControlAffineDynamics, ActorModes
from refineNCBF.utils.types import VectorBatch, MatrixBatch


@attr.dataclass
class QuadcopterVerticalParams:
    gravity: float
    mass: float
    drag_coefficient_v: float
    drag_coefficient_phi: float
    length_between_copters: float
    moment_of_inertia: float

    min_thrust: float
    max_thrust: float


default_quadcopter_vertical_params = QuadcopterVerticalParams(
    drag_coefficient_v=0.25,
    gravity=9.81,
    drag_coefficient_phi=0.02255,
    mass=2.5,
    length_between_copters=1.0,
    moment_of_inertia=1.0,
    min_thrust=0.0,
    max_thrust=0.75 * 9.81 * 2.5,  # default should be 18.39375
)


@attr.s(auto_attribs=True)
class QuadcopterVertical(ControlAffineDynamics):
    STATES = ["Y", "YDOT", "PHI", "PHIDOT"]
    CONTROLS = ["T1", "T2"]
    PERIODIC_DIMS = [2]

    gravity: float
    mass: float
    drag_coefficient_v: float
    drag_coefficient_phi: float
    length_between_copters: float
    moment_of_inertia: float

    control_lower_bounds: Tuple[float, float]
    control_upper_bounds: Tuple[float, float]

    disturbance_lower_bounds: Tuple[float] = (0.0,)
    disturbance_upper_bounds: Tuple[float] = (0.0,)

    @classmethod
    def from_specs(
        cls,
        params: QuadcopterVerticalParams,
    ) -> "QuadcopterVertical":
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
        )

    def open_loop_dynamics(self, state: VectorBatch) -> VectorBatch:
        open_loop_dynamics = np.zeros_like(state)

        open_loop_dynamics[..., 0] = state[..., 1]
        open_loop_dynamics[..., 1] = -self.drag_coefficient_v / self.mass * state[..., 1] - self.gravity
        open_loop_dynamics[..., 2] = state[..., 3]
        open_loop_dynamics[..., 3] = -self.drag_coefficient_phi / self.moment_of_inertia * state[..., 3]

        return open_loop_dynamics

    def control_matrix(self, state: VectorBatch) -> MatrixBatch:
        control_matrix = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)

        control_matrix[..., 1, :] = np.cos(state[..., 2]) / self.mass
        control_matrix[..., 3, 0] = -self.length_between_copters / self.moment_of_inertia
        control_matrix[..., 3, 1] = self.length_between_copters / self.moment_of_inertia

        return control_matrix

    def disturbance_jacobian(self, state: VectorBatch) -> MatrixBatch:
        disturbance_jacobian = np.repeat(np.zeros_like(state)[..., None], 1, axis=-1)

        return disturbance_jacobian


class QuadcopterVerticalJAX(QuadcopterVertical):
    def open_loop_dynamics(self, state: jax.Array, time: jax.Array = 0.0) -> jax.Array:
        return jnp.array(
            [
                state[1],
                -state[1] * self.drag_coefficient_v / self.mass - self.gravity,
                state[3],
                -state[3] * self.drag_coefficient_phi / self.moment_of_inertia,
            ]
        )

    def control_matrix(self, state, time: jax.Array = 0.0):
        return jnp.array(
            [
                [0, 0],
                [jnp.cos(state[2]) / self.mass, jnp.cos(state[2]) / self.mass],
                [0, 0],
                [
                    -self.length_between_copters / self.moment_of_inertia,
                    self.length_between_copters / self.moment_of_inertia,
                ],
            ]
        )

    def disturbance_jacobian(self, state: np.ndarray, time: jax.Array = 0.0) -> jax.Array:
        return jnp.expand_dims(jnp.zeros(4), axis=-1)


quadcopter_vertical_jax_hj = HJControlAffineDynamics.from_parts(
    control_affine_dynamic_system=QuadcopterVerticalJAX.from_specs(default_quadcopter_vertical_params),
    control_mode=ActorModes.MAX,
    disturbance_mode=ActorModes.MIN,
)
