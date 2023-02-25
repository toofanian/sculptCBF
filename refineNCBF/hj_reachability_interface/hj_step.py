import numpy as np

import hj_reachability
from refineNCBF.utils.types import ArrayNd, MaskNd


def hj_step(
        dynamics: hj_reachability.Dynamics,
        grid: hj_reachability.Grid,
        solver_settings: hj_reachability.SolverSettings,
        initial_values: ArrayNd,
        time_start: float,
        time_target: float,
        active_set: MaskNd = None,
        progress_bar: bool = True,
) -> ArrayNd:
    assert time_target < time_start

    return hj_reachability.step(
        solver_settings=solver_settings,
        dynamics=dynamics,
        grid=grid,
        time=time_start,
        values=initial_values,
        target_time=time_target,
        active_set=active_set,
        progress_bar=progress_bar,
    )


def hj_solve(
        dynamics: hj_reachability.Dynamics,
        grid: hj_reachability.Grid,
        solver_settings: hj_reachability.SolverSettings,
        initial_values: ArrayNd,
        time_start: float,
        time_steps: int,
        time_target: float,
        active_set: MaskNd = None,
        progress_bar: bool = True,
):
    assert time_target < time_start

    times = np.linspace(time_start, time_target, time_steps)

    return hj_reachability.solve(
        solver_settings=solver_settings,
        dynamics=dynamics,
        grid=grid,
        times=times,
        initial_values=initial_values,
        active_set=active_set,
        progress_bar=progress_bar,
    )
