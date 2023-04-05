from typing import Tuple

import attr
import jax
import jax.numpy as jnp
import numpy as np

from refineNCBF.hj_reachability_interface.hj_dynamics import HJControlAffineDynamics, ActorModes
from refineNCBF.utils.types import VectorBatch, MatrixBatch

import sys

sys.path.append("..")
from quad4d.dynamics import QuadcopterVertical


@attr.dataclass
class QuadcopterPlanarParams:
    gravity: float
    mass: float
    drag_coefficient_v: float
    drag_coefficient_phi: float
    length_between_copters: float
    moment_of_inertia: float

    min_thrust: float
    max_thrust: float


default_quadcopter_planar_params = QuadcopterPlanarParams(
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
class QuadcopterPlanar(QuadcopterVertical):
    STATES = ["Y", "YDOT", "PHI", "PHIDOT", "X", "XDOT"]
    CONTROLS = ["T1", "T2"]
    PERIODIC_DIMS = [4]

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
        params: QuadcopterPlanarParams,
    ) -> "QuadcopterPlanar":
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

    def open_loop_dynamics(self, state, time: float = 0.0):
        f = np.zeros_like(state)
        f[..., 0] = state[..., 1]
        f[..., 1] = -self.Cd_v / self.mass * state[..., 1]
        f[..., 2:] = super().open_loop_dynamics(state[..., 2:], time)
        return f

    def control_matrix(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        B = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        B[..., 1, :] = -1 / self.mass * np.sin(state[..., 4])
        B[..., 2:, :] = super().control_matrix(state[..., 2:], time)
        return B

    def state_jacobian(self, state: np.ndarray, control: np.ndarray, time: float = 0.0) -> np.ndarray:
        J = np.repeat(np.zeros_like(state)[..., None], state.shape[-1], axis=-1)
        J[..., 0, 1] = 1
        J[..., 1, 1] = -self.Cd_v / self.mass
        J[..., 1, 4] = -1 / self.mass * (control[..., 0] + control[..., 1]) * np.cos(state[..., 4])
        J[..., 2:, 2:] = super().state_jacobian(state[..., 2:], control, time)
        return J

    def disturbance_jacobian(self, state: VectorBatch, time: float = 0.0) -> MatrixBatch:
        disturbance_jacobian = np.repeat(np.zeros_like(state)[..., None], 1, axis=-1)

        return disturbance_jacobian


class QuadcopterPlanarJAX(QuadcopterPlanar):
    def open_loop_dynamics(self, state: jax.Array, time: jax.Array = 0.0) -> jax.Array:
        return jnp.array(
            [
                state[1],
                -state[1] * self.drag_coefficient_v / self.mass,
                state[3],
                -state[3] * self.drag_coefficient_v / self.mass - self.gravity,
                state[5],
                -state[5] * self.drag_coefficient_phi / self.moment_of_inertia,
            ]
        )

    def control_matrix(self, state, time: jax.Array = 0.0):
        return jnp.array(
            [
                [-jnp.sin(state[4]) / self.mass, -jnp.sin(state[4]) / self.mass],
                [0, 0],
                [jnp.cos(state[4]) / self.mass, jnp.cos(state[4]) / self.mass],
                [0, 0],
                [
                    -self.length_between_copters / self.moment_of_inertia,
                    self.length_between_copters / self.moment_of_inertia,
                ],
            ]
        )

    def disturbance_jacobian(self, state: np.ndarray, time: jax.Array = 0.0) -> jax.Array:
        return jnp.expand_dims(jnp.zeros(6), axis=-1)


quadcopter_planar_jax_hj = HJControlAffineDynamics.from_parts(
    control_affine_dynamic_system=QuadcopterPlanarJAX.from_specs(default_quadcopter_planar_params),
    control_mode=ActorModes.MAX,
    disturbance_mode=ActorModes.MIN,
)
