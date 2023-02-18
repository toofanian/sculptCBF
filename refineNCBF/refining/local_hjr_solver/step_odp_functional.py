import attr
import hj_reachability
import numpy as np
import odp.Grid
from odp.Plots import PlotOptions
from odp.solver import HJSolver

from refineNCBF.refining.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.refining.local_hjr_solver.step import LocalHjrStepper
from refineNCBF.refining.optimized_dp_interface.odp_dynamics import OdpDynamics
from refineNCBF.utils.types import MaskNd, ArrayNd


@attr.s(auto_attribs=True)
class ClassicLocalHjrStepperOdp(LocalHjrStepper):
    grid: odp.Grid.GridProcessing.Grid
    dynamics: OdpDynamics
    time_step: float
    system_objectives: dict = {"TargetSetMode": "minVWithV0"}

    @classmethod
    def from_parts(
            cls,
            dynamics: OdpDynamics,
            grid: hj_reachability.Grid,
            time_step: float,
    ):
        grid = odp.Grid.Grid(
            np.array(grid.domain.lo),
            np.array(grid.domain.hi),
            len(grid.domain.hi),
            np.array(list(grid.shape)),
            [2]
        )
        return cls(grid=grid, dynamics=dynamics, time_step=time_step)

    def __call__(self, data: LocalUpdateResult, active_set_prefiltered: MaskNd, active_set_expanded: MaskNd) -> ArrayNd:
        result = HJSolver(
            self.dynamics,
            self.grid,
            data.get_recent_values(),
            [0, -self.time_step],
            self.system_objectives,
            PlotOptions(do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 3], slicesCut=[]),
            # active_set=active_set_expanded,
            # active_set=np.ones_like(active_set_expanded, dtype=bool)
        )
        return result
