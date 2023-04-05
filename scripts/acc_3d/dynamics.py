from typing import Tuple, Optional

import attr
import jax
import jax.numpy as jnp
import numpy as np

from cbf_opt import ControlAffineDynamics
from refineNCBF.utils.types import VectorBatch, MatrixBatch


@attr.dataclass
class ActiveCruiseControlParams:
    friction_coefficients: Tuple[float, float, float]
    mass: float
    gravity: float
    target_velocity: float
    drag_coefficient: float
    th: float
    min_acceleration: float
    max_acceleration: float


default_active_cruise_control_params = ActiveCruiseControlParams(
    friction_coefficients=(0.1, 5.0, 0.25),
    mass=1650.0,
    gravity=9.81,
    target_velocity=14,
    drag_coefficient=0.3,
    th=1.8,
    min_acceleration=-4855.95,
    max_acceleration=4855.95,
)

simplified_active_cruise_control_params = ActiveCruiseControlParams(
    friction_coefficients=(0, 0, 0),
    mass=1650.0,
    gravity=9.81,
    target_velocity=0.0,
    drag_coefficient=0.3,
    th=1.8,
    min_acceleration=-4855.95,
    max_acceleration=4855.95,
)


@attr.s(auto_attribs=True)
class ActiveCruiseControl(ControlAffineDynamics):
    STATES = ["Distance", "Velocity", "Delta Distance"]
    CONTROLS = ["Acceleration"]

    friction_coefficients: Tuple[float, float, float]
    mass: float
    gravity: float
    target_velocity: float
    drag_coefficient: float
    th: float

    n_dims: int = 3
    periodic_state_dimensions: Tuple[float, ...] = ()

    control_dims: int = 1

    disturbance_dims: int = 1
    disturbance_lower_bounds: Tuple[float, ...] = (0,)
    disturbance_upper_bounds: Tuple[float, ...] = (0,)

    @classmethod
    def from_params(cls, params: ActiveCruiseControlParams):
        control_lower_bounds = (params.min_acceleration,)
        control_upper_bounds = (params.max_acceleration,)
        return cls(
            friction_coefficients=params.friction_coefficients,
            mass=params.mass,
            gravity=params.gravity,
            target_velocity=params.target_velocity,
            drag_coefficient=params.drag_coefficient,
            th=params.th,
            control_lower_bounds=control_lower_bounds,
            control_upper_bounds=control_upper_bounds,
        )

    @classmethod
    def from_specs(
        cls,
        friction_coefficients: Tuple[float, float, float] = (0.1, 5.0, 0.25),
        mass: float = 1650.0,
        gravity: float = 9.81,
        target_velocity: float = 14.0,
        drag_coefficient: float = 0.3,
        min_acceleration: Optional[float] = None,
        max_acceleration: Optional[float] = None,
    ) -> "ActiveCruiseControl":

        if min_acceleration is None:
            min_acceleration = -drag_coefficient * gravity * mass
        if max_acceleration is None:
            max_acceleration = drag_coefficient * gravity * mass

        assert min_acceleration <= max_acceleration
        assert friction_coefficients[0] >= 0
        assert friction_coefficients[1] >= 0
        assert friction_coefficients[2] >= 0
        assert mass > 0
        assert gravity > 0
        assert target_velocity > 0
        assert drag_coefficient >= 0

        control_lower_bounds = (min_acceleration,)
        control_upper_bounds = (max_acceleration,)

        return cls(
            friction_coefficients=friction_coefficients,
            mass=mass,
            gravity=gravity,
            target_velocity=target_velocity,
            drag_coefficient=drag_coefficient,
            control_lower_bounds=control_lower_bounds,
            control_upper_bounds=control_upper_bounds,
        )

    def _get_rolling_resistance(self, state: VectorBatch) -> VectorBatch:
        return (
            self.friction_coefficients[0]
            + self.friction_coefficients[1] * state[..., 1]
            + self.friction_coefficients[2] * state[..., 1] ** 2
        )

    def open_loop_dynamics(self, state: VectorBatch) -> VectorBatch:
        open_loop_dynamics = np.zeros_like(state)
        open_loop_dynamics[..., 0] = state[..., 1]
        open_loop_dynamics[..., 1] = -1 / self.mass * self._get_rolling_resistance(state)
        open_loop_dynamics[..., 2] = self.target_velocity - state[..., 1]
        return open_loop_dynamics

    def control_matrix(self, state: VectorBatch) -> MatrixBatch:
        control_jacobian = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        control_jacobian[..., 1, 0] = 1 / self.mass
        return control_jacobian

    def disturbance_jacobian(
        self,
        state: VectorBatch,
    ) -> MatrixBatch:
        disturbance_jacobian = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        return disturbance_jacobian


@attr.s(auto_attribs=True, eq=False)
class ActiveCruiseControlJAX(ActiveCruiseControl):
    def _get_rolling_resistance(self, state: jax.Array) -> jax.Array:
        rolling_resistance = (
            self.friction_coefficients[0]
            + self.friction_coefficients[1] * state[1]
            + self.friction_coefficients[2] * state[1] ** 2
        )
        return rolling_resistance

    def open_loop_dynamics(self, state, time=0.0):
        return jnp.array(
            [state[1], -1 / self.mass * self._get_rolling_resistance(state), self.target_velocity - state[1]]
        )

    def control_matrix(self, state, time=0.0):
        return jnp.expand_dims(jnp.array([0, 1 / self.mass, 0]), axis=-1)

    def disturbance_jacobian(self, state, time=0.0):
        return jnp.expand_dims(jnp.zeros(3), axis=-1)
