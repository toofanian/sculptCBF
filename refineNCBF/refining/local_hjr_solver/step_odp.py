from typing import Optional

import attr
import hj_reachability
import jax
import numpy as np
import odp.Grid
from odp.Plots import PlotOptions
from odp.solver import HjSolver

from refineNCBF.refining.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.refining.local_hjr_solver.step_hj import LocalHjrStepper
from refineNCBF.refining.optimized_dp_interface.odp_dynamics import OdpDynamics
from refineNCBF.utils.sets import compute_signed_distance
from refineNCBF.utils.types import MaskNd, ArrayNd

#
# @attr.s(auto_attribs=True)
# class ClassicLocalHjrStepperOdp(LocalHjrStepper):
#     grid: odp.Grid.GridProcessing.Grid
#     dynamics: OdpDynamics
#     time_step: float
#     system_objectives: dict = {"TargetSetMode": "minVWithV0"}
#
#     @classmethod
#     def from_parts(
#             cls,
#             dynamics: OdpDynamics,
#             grid: hj_reachability.Grid,
#             time_step: float,
#     ):
#         grid = odp.Grid.Grid(
#             np.array(grid.domain.lo),
#             np.array(grid.domain.hi),
#             len(grid.domain.hi),
#             np.array(list(grid.shape)),
#             [2]
#         )
#         return cls(grid=grid, dynamics=dynamics, time_step=time_step)
#
#     def __call__(self, data: LocalUpdateResult, active_set_prefiltered: MaskNd, active_set_expanded: MaskNd) -> ArrayNd:
#         values = data.get_recent_values()
#         result = self.hj_solver(
#             self.dynamics,
#             self.grid,
#             values,
#             [0, -self.time_step],
#             self.system_objectives,
#             PlotOptions(do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 3], slicesCut=[]),
#             active_set=active_set_expanded,
#         )
#         return result


@attr.s(auto_attribs=True)
class DecreaseLocalHjrStepperOdp(LocalHjrStepper):
    grid: odp.Grid.GridProcessing.Grid
    dynamics: OdpDynamics
    time_step: float
    system_objectives: dict
    hj_solver: HjSolver

    @classmethod
    def from_parts(
            cls,
            dynamics: OdpDynamics,
            grid: hj_reachability.Grid,
            time_step: float,
    ):
        grid_odp = odp.Grid.Grid(
            np.array(grid.domain.lo),
            np.array(grid.domain.hi),
            len(grid.domain.hi),
            np.array(list(grid.shape)),
            [2]
        )
        system_objectives = {"TargetSetMode": "minVWithV0"}
        hj_solver = HjSolver.from_parts(
            dynamics,
            grid_odp,
            system_objectives
        )
        return cls(grid=grid_odp, dynamics=dynamics, time_step=time_step, system_objectives=system_objectives, hj_solver=hj_solver)

    def __call__(self, data: LocalUpdateResult, active_set_prefiltered: MaskNd, active_set_expanded: MaskNd) -> ArrayNd:
        values = data.get_recent_values()
        next_result = self.hj_solver(
            self.dynamics,
            self.grid,
            values,
            [0, -self.time_step],
            self.system_objectives,
            PlotOptions(do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 3], slicesCut=[]),
            active_set=active_set_expanded
        )
        next_result = jax.numpy.array(next_result)
        where_decrease = (next_result < values)
        thing = jax.numpy.array(values)
        thing = thing.at[where_decrease].set(next_result[where_decrease])
        return thing
