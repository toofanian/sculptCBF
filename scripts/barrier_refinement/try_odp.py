import hj_reachability
import jax
import numpy as np
from odp.Plots import PlotOptions
from odp.solver import hj_solve
import odp.Grid
import odp

from refineNCBF.dynamic_systems.implementations.active_cruise_control_odp import ActiveCruiseControlOdp
from refineNCBF.refining.local_hjr_solver.expand import SignedDistanceNeighborsNearBoundary
from refineNCBF.refining.local_hjr_solver.prefilter import PreFilterWhereFarFromBoundarySplit
from refineNCBF.refining.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.utils.sets import compute_signed_distance, get_mask_boundary_on_both_sides_by_signed_distance


def demo():
    dynamics = ActiveCruiseControlOdp()

    grid = hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj_reachability.sets.Box(
            [0, -20, 20],
            [1e3, 20, 80]
        ),
        shape=(3, 301, 301)
    )

    avoid_set = (
            (grid.states[..., 2] > 60)
            |
            (grid.states[..., 2] < 40)
    )

    reach_set = jax.numpy.zeros_like(avoid_set, dtype=bool)

    terminal_values = compute_signed_distance(~avoid_set)

    active_set_pre_filter = PreFilterWhereFarFromBoundarySplit.from_parts(
            distance_inner=2,
            distance_outer=2,
    )

    neighbor_expander = SignedDistanceNeighborsNearBoundary.from_parts(
        neighbor_distance=2,
        boundary_distance_inner=2,
        boundary_distance_outer=2,
    )

    active_set = get_mask_boundary_on_both_sides_by_signed_distance(~avoid_set, distance=2)
    initial_values = terminal_values.copy()

    data = LocalUpdateResult.from_parts(
                    local_solver=None,
                    dynamics=dynamics,
                    grid=grid,
                    avoid_set=avoid_set,
                    reach_set=reach_set,
                    seed_set=active_set,
                    initial_values=initial_values,
                    terminal_values=terminal_values
    )

    grid_odp = odp.Grid.Grid(
        np.array(grid.domain.lo),
        np.array(grid.domain.hi),
        len(grid.domain.hi),
        np.array(list(grid.shape)),
        [2]
    )

    result = hj_solve(
        dynamics,
        grid_odp,
        initial_values,
        [0, 1],
        {"TargetSetMode": "minVWithV0"},
        PlotOptions(do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 3], slicesCut=[]),
    )


if __name__ == '__main__':
    demo()
