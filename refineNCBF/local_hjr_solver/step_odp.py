from typing import List

import attr
import jax
import numpy as np

import hj_reachability
import odp.Grid
from odp.Plots import PlotOptions
from odp.solver import HJSolverClass
from refineNCBF.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.local_hjr_solver.step_hj import LocalHjrStepper
from refineNCBF.local_hjr_solver.step_odp_type import OdpStepper
from refineNCBF.optimized_dp_interface.odp_dynamics import OdpDynamics
from refineNCBF.utils.types import MaskNd, ArrayNd


@attr.s(auto_attribs=True)
class ClassicLocalHjrStepperOdp(LocalHjrStepper, OdpStepper):
    grid: odp.Grid.GridProcessing.Grid
    dynamics: OdpDynamics
    time_step: float
    system_objectives: dict
    integration_scheme: str
    hj_solver: HJSolverClass
    global_minimizing: bool

    @classmethod
    def from_parts(
        cls,
        dynamics: OdpDynamics,
        grid: hj_reachability.Grid,
        periodic_dims: List[int],
        integration_scheme: str,
        time_step: float,
        global_minimizing: bool = False,
    ):
        grid_odp = odp.Grid.Grid(
            np.array(grid.domain.lo),
            np.array(grid.domain.hi),
            len(grid.domain.hi),
            np.array(list(grid.shape)),
            periodic_dims,
        )
        system_objectives = {"TargetSetMode": "minVWithV0"}
        hj_solver = HJSolverClass()
        return cls(
            grid=grid_odp,
            dynamics=dynamics,
            time_step=time_step,
            system_objectives=system_objectives,
            integration_scheme=integration_scheme,
            hj_solver=hj_solver,
            global_minimizing=global_minimizing,
        )

    def __call__(self, data: LocalUpdateResult, active_set_prefiltered: MaskNd, active_set_expanded: MaskNd) -> ArrayNd:
        values = data.get_recent_values()
        next_result = self.hj_solver(
            self.dynamics,
            self.grid,
            values,
            [0, -self.time_step],
            self.system_objectives,
            PlotOptions(do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 3], slicesCut=[]),
            accuracy="medium",
            int_scheme=self.integration_scheme,
            active_set=active_set_expanded,
            verbose=False,
            untilConvergent=True,
            global_minimizing=self.global_minimizing,
        )
        return next_result


@attr.s(auto_attribs=True)
class DecreaseLocalHjrStepperOdp(LocalHjrStepper, OdpStepper):
    grid: odp.Grid.GridProcessing.Grid
    dynamics: OdpDynamics
    time_step: float
    system_objectives: dict
    integration_scheme: str
    global_minimizing: bool
    hj_solver: HJSolverClass

    @classmethod
    def from_parts(
        cls,
        dynamics: OdpDynamics,
        grid: hj_reachability.Grid,
        periodic_dims: List[int],
        integration_scheme: str,
        time_step: float,
        global_minimizing: bool = False,
    ):
        grid_odp = odp.Grid.Grid(
            np.array(grid.domain.lo),
            np.array(grid.domain.hi),
            len(grid.domain.hi),
            np.array(list(grid.shape)),
            periodic_dims,
        )
        system_objectives = {"TargetSetMode": "minVWithV0"}
        hj_solver = HJSolverClass()
        return cls(
            grid=grid_odp,
            dynamics=dynamics,
            time_step=time_step,
            system_objectives=system_objectives,
            integration_scheme=integration_scheme,
            hj_solver=hj_solver,
            global_minimizing=global_minimizing,
        )

    def __call__(self, data: LocalUpdateResult, active_set_prefiltered: MaskNd, active_set_expanded: MaskNd) -> ArrayNd:
        values = data.get_recent_values()
        next_result = self.hj_solver(
            self.dynamics,
            self.grid,
            values,
            [0, -self.time_step],
            self.system_objectives,
            PlotOptions(do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 3], slicesCut=[]),
            accuracy="medium",
            int_scheme=self.integration_scheme,
            active_set=active_set_expanded,
            verbose=False,
            global_minimizing=self.global_minimizing,
        )
        next_result = jax.numpy.array(next_result)
        where_decrease = next_result < (values - 1e-3)
        # print(jax.numpy.count_nonzero(where_decrease), jax.numpy.count_nonzero(active_set_expanded))

        # NOTE: there is minor solver noise that requires us to redundantly filter on active_set_expanded. otherwise
        #  over each iteration we will introduce a very small negative drift to the value function
        where_decrease_and_active = where_decrease & active_set_expanded
        thing = jax.numpy.array(values)
        thing = thing.at[where_decrease_and_active].set(next_result[where_decrease_and_active])
        return thing
