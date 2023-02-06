import attr

from refineNCBF.dynamic_systems.implementations.active_cruise_control import ActiveCruiseControl
from refineNCBF.refining.hj_reachability_interface.hj_dynamics import HJControlAffineDynamics


@attr.s(auto_attribs=True)
class OdpDynamics(HJControlAffineDynamics):
    def opt_ctrl(self, t, state, spat_deriv):
        ...

    def opt_dstr(self, t, state, spat_deriv):
        ...

    def dynamics(self, t, state, uOpt, dOpt):
        ...

