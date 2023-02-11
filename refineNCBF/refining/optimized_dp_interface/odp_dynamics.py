from abc import ABC, abstractmethod

import attr


@attr.s(auto_attribs=True)
class OdpDynamics(ABC):
    @abstractmethod
    def dynamics(self, t, state, uOpt, dOpt):
        ...

    @abstractmethod
    def opt_ctrl(self, t, state, spat_deriv):
        ...

    @abstractmethod
    def opt_dstb(self, t, state, spat_deriv):
        ...
