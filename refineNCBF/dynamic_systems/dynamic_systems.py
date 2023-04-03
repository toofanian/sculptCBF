from abc import ABC, abstractmethod
from typing import Tuple

import attr
import numpy as np

from refineNCBF.utils.types import VectorBatch, MatrixBatch


@attr.s(auto_attribs=True, eq=False)
class DynamicSystem(ABC):
    state_dimensions: int
    periodic_state_dimensions: Tuple[int, ...]

    control_dimensions: int
    control_lower_bounds: Tuple[float, ...]
    control_upper_bounds: Tuple[float, ...]

    disturbance_dimensions: int
    disturbance_lower_bounds: Tuple[float, ...]
    disturbance_upper_bounds: Tuple[float, ...]

    def __init__(self, **kwargs):
        self.n_dims = self.state_dimensions
        self.controls_dims = self.control_dimensions
        self.disturbance_dims = self.disturbance_dimensions
        self.periodic_dims = self.periodic_state_dimensions

    def __call__(self, state, control, time):
        return self.compute_dynamics(state, control, np.zeros((state.shape[0], self.disturbance_dims)))

    def wrap_state(self, state: VectorBatch) -> VectorBatch:
        """

        :param state: (batch_size, state_dimensions)
        :return:
        """
        wrapped_state = np.zeros_like(state)
        for periodic_dimension in self.periodic_state_dimensions:
            wrapped_state[..., periodic_dimension] = np.remainder(
                state[..., periodic_dimension],
                2 * np.pi,
            )
        return wrapped_state

    def wrap_dynamics(self, state: np.ndarray) -> np.ndarray:
        """
        Periodic dimensions are wrapped to [-pi, pi)
        Args:
            state (np.ndarray): Unwrapped state
        Returns:
            state (np.ndarray): Wrapped state
        """
        for periodic_dim in self.periodic_dims:
            try:
                state[..., periodic_dim] = (state[..., periodic_dim] + np.pi) % (2 * np.pi) - np.pi
            except TypeError:  # FIXME: Clunky at best, how to deal with jnp and np mix
                state = state.at[periodic_dim].set((state[periodic_dim] + np.pi) % (2 * np.pi) - np.pi)

        return state

    def step(self, state: np.ndarray, control: np.ndarray, time: float = 0.0, scheme: str = "fe") -> np.ndarray:
        """Implements the discrete-time dynamics ODE
        scheme in {fe, rk4}"""
        if scheme == "fe":
            n_state = state + self(state, control, time) * self.dt
        elif scheme == "rk4":
            # TODO: Figure out how to do RK4 with periodic dimensions (aka angle normalization)
            # Assumes zoh on control
            k1 = self(state, control, time)
            k2 = self(state + k1 * self.dt / 2, control, time + self.dt / 2)
            k3 = self(state + k2 * self.dt / 2, control, time + self.dt / 2)
            k4 = self(state + k3 * self.dt, control, time + self.dt)
            n_state = state + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6
        else:
            raise ValueError("scheme must be either 'fe' or 'rk4'")
        return self.wrap_dynamics(n_state)

    def compute_dynamics(self, state: VectorBatch, control: VectorBatch, disturbance: VectorBatch) -> VectorBatch:
        assert state.shape[1] == self.state_dimensions
        assert control.shape[1] == self.control_dimensions
        assert disturbance.shape[1] == self.disturbance_dimensions
        assert state.shape[0] == control.shape[0] == disturbance.shape[0]
        return self._compute_dynamics(state, control, disturbance)

    @abstractmethod
    def _compute_dynamics(
        self,
        state: VectorBatch,
        control: VectorBatch,
        disturbance: VectorBatch,
    ) -> VectorBatch:
        ...


@attr.s(auto_attribs=True, eq=False)
class ControlAffineDynamicSystem(DynamicSystem, ABC):
    def _compute_dynamics(
        self,
        state: VectorBatch,
        control: VectorBatch,
        disturbance: VectorBatch,
    ) -> VectorBatch:
        open_loop_dynamics = self.compute_open_loop_dynamics(state)
        control_jacobian = self.compute_control_jacobian(state)
        disturbance_jacobian = self.compute_disturbance_jacobian(state)

        # TODO verify that this is correct for batch
        dynamics = open_loop_dynamics + control_jacobian @ control + disturbance_jacobian @ disturbance
        return dynamics

    def open_loop_dynamics(self, state: np.ndarray, time: float = 0.0):
        return self.compute_open_loop_dynamics(state)

    @abstractmethod
    def compute_open_loop_dynamics(
        self,
        state: VectorBatch,
    ) -> VectorBatch:
        ...

    def control_jacobian(self, state: np.ndarray, time: float = 0.0):
        return self.control_matrix(state, time)

    def control_matrix(self, state: np.ndarray, time: float = 0.0):
        return self.compute_control_jacobian(state)

    def disturbance_jacobian(self, state: np.ndarray, time: float = 0.0):
        return self.compute_disturbance_jacobian(state)

    @abstractmethod
    def compute_control_jacobian(
        self,
        state: VectorBatch,
    ) -> MatrixBatch:
        ...

    @abstractmethod
    def compute_disturbance_jacobian(
        self,
        state: VectorBatch,
    ) -> MatrixBatch:
        ...


@attr.s(auto_attribs=True, eq=False)
class ControlAffineDynamicSystemFixedPolicy(ControlAffineDynamicSystem, ABC):
    @abstractmethod
    def compute_control(self, state: VectorBatch) -> VectorBatch:
        ...

    @abstractmethod
    def compute_disturbance(self, state: VectorBatch) -> VectorBatch:
        ...
