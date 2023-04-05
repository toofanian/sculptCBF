from copy import copy
from typing import List, Tuple
from cbf_opt.dynamics import Dynamics
from matplotlib.figure import Figure
import pandas as pd
from cbf_opt import ControlAffineDynamics
import numpy as np
import matplotlib.pyplot as plt
from refineNCBF.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.utils.visuals import ArraySlice2D, DimName
from refine_cbfs.cbf import TabularControlAffineCBF
from experiment_wrapper import RolloutTrajectory, TimeSeriesExperiment, StateSpaceExperiment
import seaborn as sns
from validate_barrier.visualize_quad4d import SafevUnsafeStateSpaceExperiment
from dynamics_interface import QuadPlanarDynamicsInterface

# file_name = "230317_0024"
# file_name = "230317_0644_full"
if __name__ == "__main__":
    file_name = "230318_2346"
    import os

    # Get directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, "{}.csv".format(file_name)))

    local_hjr_dict = {
        # "global_jax": "_20230307_191036",
        # "local_jax": "20230307_025911",
        "local_odp": "20230307_180336",
        "global_odp": "_20230307_183541",
    }

    cbfs = {}
    converged_cbvfs = {}
    initial_cbfs = {}
    dynamics_instance = QuadPlanarDynamicsInterface()
    for i, (key, value) in enumerate(local_hjr_dict.items()):
        patching_result = LocalUpdateResult.load("/data/6dim/{}.dill".format(value))
        if i == 0:
            grid = patching_result.grid
        else:
            assert (grid.states == patching_result.grid.states).all()
        result_cbf_values = patching_result.iterations[-1].computed_values
        converged_cbvf = TabularControlAffineCBF(dynamics_instance.dynamics, {}, grid=grid)
        converged_cbvf.vf_table = result_cbf_values
        cbfs[key] = converged_cbvf
        converged_cbvfs[key] = result_cbf_values
        initial_cbfs[key] = patching_result.terminal_values

    reference_index = patching_result.get_middle_index()

    ref_index = ArraySlice2D.from_reference_index(
        reference_index=reference_index,
        # reference_index=(0, 7, 0, 7, 30, 7),
        # free_dim_1=DimName(0, 'x'),
        free_dim_2=DimName(0, "y"),
        free_dim_1=DimName(1, "ydot"),
    )

    reference_slice = ref_index
    x1, x2 = np.meshgrid(
        grid.coordinate_vectors[reference_slice.free_dim_1.dim],
        grid.coordinate_vectors[reference_slice.free_dim_2.dim],
    )

    for exp in df.controller.unique():
        df_new = df[df.controller == exp]
        # time_series_rollout = SafevUnsafeTimeSeriesExperiment("TS", start_x=0, x_indices=[0])

        # fig_handle = time_series_rollout.plot(dynamics, df_new, color=["green"], alpha=[0.2])
        # fig = fig_handle[0][1]
        # ax = fig.axes[0]
        # fig_handle[0][1].savefig(os.path.join(dir_path, "{}_ts_{}.png".format(file_name, exp)))
        # Remove neural or lqr from end of string
        if "neural" in exp:
            exp2 = exp[:-6]
        elif "lqr" in exp:
            exp2 = exp[:-3]
        try:
            resulting_cbf_values = converged_cbvfs[exp2]
            result_cbf_init_values = initial_cbfs[exp2]
        except KeyError:
            resulting_cbf_values = initial_cbfs["global_jax"]
            result_cbf_init_values = initial_cbfs["global_jax"]
        state_space_rollout = SafevUnsafeStateSpaceExperiment("Rollout", start_x=0, x_indices=[1, 0])
        fig_handle = state_space_rollout.plot(
            dynamics_instance.dynamics, df_new, add_direction=False, color=["green"], alpha=[0.2]
        )
        fig = fig_handle[0][1]
        ax = fig.axes[0]

        ax.contour(
            x1,
            x2,
            reference_slice.get_sliced_array(resulting_cbf_values).T,
            levels=[0],
            colors="darkgrey",
            linewidths=4,
        )
        # ax.contour(x1, x2, other_slice, levels=[0], colors="k")

        ax.contour(
            x1,
            x2,
            reference_slice.get_sliced_array(result_cbf_init_values).T,
            levels=[0],
            colors="lightgrey",
            linewidths=4,
            linestyles="dashed",
        )
        # ax.contour(x1, x2, original_slice, levels=[0], colors="orange")
        ax.set_xlim(-8.0, 8.0)
        ax.set_ylim(0, 10)
        fig_handle[0][1].savefig(os.path.join(dir_path, "{}_{}.png".format(file_name, exp)))
        # # get axis of fig
