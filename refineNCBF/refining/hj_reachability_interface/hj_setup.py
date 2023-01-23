from typing import Union

import attr
import hj_reachability


@attr.s(auto_attribs=True, frozen=True, eq=False)
class HjSetup:
    dynamics: Union[hj_reachability.Dynamics, hj_reachability.ControlAndDisturbanceAffineDynamics]
    grid: hj_reachability.Grid

    @classmethod
    def from_parts(
            cls,
            dynamics: hj_reachability.Dynamics,
            grid: hj_reachability.Grid,
    ) -> 'HjSetup':
        return cls(dynamics=dynamics, grid=grid)
