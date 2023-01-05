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

    # TODO: wrap state is unused, since there is no "step" function which returns xdot*t + x, just dynamics which returns xdot
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

    def compute_dynamics(
            self,
            state: VectorBatch,
            control: VectorBatch,
            disturbance: VectorBatch
    ) -> VectorBatch:
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

    @abstractmethod
    def compute_open_loop_dynamics(
            self,
            state: VectorBatch,
    ) -> VectorBatch:
        ...

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
