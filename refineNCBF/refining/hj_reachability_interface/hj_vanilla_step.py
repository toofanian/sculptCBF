import hj_reachability

from refineNCBF.refining.hj_reachability_interface.hj_setup import HjSetup
from refineNCBF.utils.types import ArrayNd


def hjr_solve_vanilla(
        hj_setup: HjSetup,
        solver_settings: hj_reachability.SolverSettings,
        initial_values: ArrayNd,
        time_start: float,
        time_target: float,
        progress_bar: bool = True,
) -> ArrayNd:
    return hj_reachability.step(
        solver_settings=solver_settings,
        dynamics=hj_setup.dynamics,
        grid=hj_setup.grid,
        time=time_start,
        values=initial_values,
        target_time=time_target,
        progress_bar=progress_bar,
    )
