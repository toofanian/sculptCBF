from enum import IntEnum

import attr
import hj_reachability
from jax import numpy as jnp

from refineNCBF.dynamic_systems.dynamic_systems import ControlAffineDynamicSystem


class ActorModes(IntEnum):
    MIN = 0
    MAX = 1


@attr.s(auto_attribs=True, eq=False)
class HJControlAffineDynamics(hj_reachability.ControlAndDisturbanceAffineDynamics):
    control_affine_dynamic_system: ControlAffineDynamicSystem

    control_mode: str
    disturbance_mode: str
    control_space: hj_reachability.sets.Box
    disturbance_space: hj_reachability.sets.Box

    @classmethod
    def from_control_affine_dynamics(
            cls,
            control_affine_dynamic_system: ControlAffineDynamicSystem,
            control_mode: ActorModes,
            disturbance_mode: ActorModes,
    ):
        control_space = hj_reachability.sets.Box(
            jnp.array(control_affine_dynamic_system.control_lower_bounds),
            jnp.array(control_affine_dynamic_system.control_upper_bounds)
        )

        disturbance_space = hj_reachability.sets.Box(
            jnp.array(control_affine_dynamic_system.disturbance_lower_bounds),
            jnp.array(control_affine_dynamic_system.disturbance_upper_bounds)
        )

        if control_mode == ActorModes.MIN:
            control_mode = 'min'
        else:
            control_mode = 'max'

        if disturbance_mode == ActorModes.MIN:
            disturbance_mode = 'min'
        else:
            disturbance_mode = 'max'

        return cls(
            control_affine_dynamic_system=control_affine_dynamic_system,
            control_mode=control_mode,
            disturbance_mode=disturbance_mode,
            control_space=control_space,
            disturbance_space=disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        return self.control_affine_dynamic_system.compute_open_loop_dynamics(state=state)

    def control_jacobian(self, state, time):
        return self.control_affine_dynamic_system.compute_control_jacobian(state=state)

    def disturbance_jacobian(self, state, time):
        return self.control_affine_dynamic_system.compute_disturbance_jacobian(state=state)
