import hj_reachability

from refineNCBF.refining.hj_reachability_interface.hj_setup import HjSetup
from refineNCBF.utils.types import ArrayNd, MaskNd


def hj_step(
        hj_setup: HjSetup,
        solver_settings: hj_reachability.SolverSettings,
        initial_values: ArrayNd,
        time_start: float,
        time_target: float,
        active_set: MaskNd = None,
        progress_bar: bool = True,
) -> ArrayNd:
    return hj_reachability.step(
        solver_settings=solver_settings,
        dynamics=hj_setup.dynamics,
        grid=hj_setup.grid,
        time=time_start,
        values=initial_values,
        target_time=time_target,
        active_set=active_set,
        progress_bar=progress_bar,
    )
