import attr
import hj_reachability


@attr.s(auto_attribs=True, frozen=True, eq=False)
class HjSetup:
    dynamics: hj_reachability.Dynamics
    grid: hj_reachability.Grid

    @classmethod
    def from_parts(
            cls,
            dynamics: hj_reachability.Dynamics,
            grid: hj_reachability.Grid,
    ):
        return cls(dynamics=dynamics, grid=grid)
