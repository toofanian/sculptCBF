from typing import Tuple

import attr
import jax

from refineNCBF.dynamic_systems.dynamic_systems import ControlAffineDynamicSystem
from refineNCBF.refining.hj_reachability_interface.hj_dynamics import HJControlAffineDynamics, ActorModes
from refineNCBF.utils.types import Matrix, Vector


@attr.s(auto_attribs=True)
class GoLeftJax(ControlAffineDynamicSystem):
    state_dimensions: int = 1
    periodic_state_dimensions: Tuple[float, ...] = ()

    control_dimensions: int = 1
    control_lower_bounds: Tuple[float, ...] = (0,)
    control_upper_bounds: Tuple[float, ...] = (0,)

    disturbance_dimensions: int = 1
    disturbance_lower_bounds: Tuple[float, ...] = (0,)
    disturbance_upper_bounds: Tuple[float, ...] = (0,)

    def compute_open_loop_dynamics(
            self,
            state: Vector,
    ) -> Vector:
        return jax.numpy.array([-1])

    def compute_control_jacobian(
            self,
            state: Vector,
    ) -> Matrix:
        return jax.numpy.array([[0]])

    def compute_disturbance_jacobian(
            self,
            state: Vector,
    ) -> Matrix:
        return jax.numpy.array([[0]])


go_left_jax_hj = HJControlAffineDynamics.from_parts(
    control_affine_dynamic_system=GoLeftJax(),
    control_mode=ActorModes.MAX,
    disturbance_mode=ActorModes.MIN
)
