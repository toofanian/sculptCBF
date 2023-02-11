import attr

from refineNCBF.dynamic_systems.implementations.active_cruise_control import ActiveCruiseControl
from refineNCBF.refining.optimized_dp_interface.odp_dynamics import OdpDynamics

import heterocl as hcl


@attr.s(auto_attribs=True)
class ActiveCruiseControlOdp(OdpDynamics, ActiveCruiseControl):
    def opt_ctrl(self, t, state, spat_deriv):
        opt_a = hcl.scalar(self.control_upper_bounds[0], "opt_a")
        in2 = hcl.scalar(0, "in2")
        in3 = hcl.scalar(0, "in3")

        with hcl.if_(spat_deriv[1] < 0):
            opt_a[0] = -opt_a

        return opt_a[0], in2[0], in3[0]

    def opt_dstb(self, t, state, spat_deriv):
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        return d1[0], d2[0], d3[0]

    def dynamics(self, t, state, uOpt, dOpt):
        x1_dot = hcl.scalar(0, "x1_dot")
        x2_dot = hcl.scalar(0, "x2_dot")
        x3_dot = hcl.scalar(0, "x3_dot")

        x1_dot[0] = state[1]
        x2_dot[0] = -1 / self.mass * \
            (self.friction_coefficients[0] + self.friction_coefficients[1] * state[1] + self.friction_coefficients[2] * state[1] ** 2) \
            + uOpt[0]
        x3_dot[0] = self.target_velocity - state[1]
        return x1_dot[0], x2_dot[0], x3_dot[0]
