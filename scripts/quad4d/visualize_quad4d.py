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


class SafevUnsafeStateSpaceExperiment(StateSpaceExperiment):
    def plot(
        self, dynamics: Dynamics, results_df: pd.DataFrame, display_plots: bool = False, **kwargs
    ) -> List[Tuple[str, Figure]]:
        """Overrides Experiment.plot to plot state space data. Same args as Experiment.plot"""
        self.set_idx_and_labels(dynamics)
        assert len(self.x_labels) in [2, 3], "Can't plot in this dimension!"

        ax = kwargs.get("ax")
        alpha = kwargs.get("alpha", [1] * len(results_df.controller.unique()))
        linestyles = kwargs.get("linestyles", ["-"] * len(results_df.controller.unique()))
        # 2D visualization
        if len(self.x_labels) == 2:

            if ax is None:
                fig, ax = plt.subplots()
                fig.set_size_inches(9, 6)
            else:
                fig = ax.get_figure()
            i = -1
            for controller in results_df.controller.unique():
                i += 1
                for scenario in results_df.scenario.unique():
                    for rollout in results_df.rollout.unique():
                        mask = (
                            (results_df.controller == controller)
                            & (results_df.scenario == scenario)
                            & (results_df.rollout == rollout)
                        )
                        xmask = mask & (results_df.measurement.values == self.x_labels[0])
                        ymask = mask & (results_df.measurement.values == self.x_labels[1])
                        xvals = results_df[xmask].value.values
                        yvals = results_df[ymask].value.values
                        if not results_df[xmask].unsafe.isna().all():
                            safexmask = (
                                mask
                                & (results_df.measurement.values == self.x_labels[0])
                                & (results_df.unsafe.values == False)
                            )
                            safeymask = (
                                mask
                                & (results_df.measurement.values == self.x_labels[1])
                                & (results_df.unsafe.values == False)
                            )
                            unsafexmask = (
                                mask
                                & (results_df.measurement.values == self.x_labels[0])
                                & (results_df.unsafe.values == True)
                            )
                            unsafeymask = (
                                mask
                                & (results_df.measurement.values == self.x_labels[1])
                                & (results_df.unsafe.values == True)
                            )
                            safexvals = results_df[safexmask].value.values
                            safeyvals = results_df[safeymask].value.values
                            unsafexvals = results_df[unsafexmask].value.values
                            unsafeyvals = results_df[unsafeymask].value.values
                            l = ax.plot(safexvals, safeyvals, ".", alpha=alpha[i], color="green", markersize=2.0)
                            ax.plot(unsafexvals, unsafeyvals, ".", alpha=alpha[i] * 1.6, color="red", markersize=2.0)
                            ax.plot(
                                xvals[0],
                                yvals[0],
                                "o",
                                color=l[0].get_color(),
                                alpha=alpha[i],
                                linewidth=0.5,
                                markersize=0.2,
                            )
                            ax.plot(
                                xvals[-1],
                                yvals[-1],
                                "x",
                                color=l[0].get_color(),
                                alpha=alpha[i],
                                linewidth=0.5,
                                markersize=0.2,
                            )
                        else:
                            if kwargs.get("color") is None:
                                l = ax.plot(xvals, yvals, alpha=alpha[i])
                            else:
                                l = ax.plot(
                                    xvals,
                                    yvals,
                                    color=kwargs.get("color")[i],
                                    alpha=alpha[i],
                                    ls=linestyles[i],
                                )
                            ax.plot(xvals[0], yvals[0], "o", color=l[0].get_color(), alpha=alpha[i])
                            ax.plot(xvals[-1], yvals[-1], "x", color=l[0].get_color(), alpha=alpha[i])
            # ax.set_xlabel(self.x_labels[0])
            # ax.set_ylabel(self.x_labels[1])
        # 3D visualization
        else:
            raise NotImplementedError("Future work!")

        fig_handle = ("State space visualization", fig)
        return [fig_handle]


class SafevUnsafeTimeSeriesExperiment(TimeSeriesExperiment):
    def plot(
        self,
        dynamics: Dynamics,
        results_df: pd.DataFrame,
        extra_measurements: list = [],
        display_plots: bool = False,
        **kwargs
    ) -> List[Tuple[str, Figure]]:
        """Overrides Experiment.plot to plot the time series of the measurements. Same args as Experiment.plot, but also:

        Extra Args:
            extra_measurements (list, optional): other variables (beyond x_labels and y_labels to display).
        """
        self.set_idx_and_labels(dynamics)
        sns.set_theme(context="talk", style="white")
        default_colors = sns.color_palette("colorblind")
        colors = kwargs.get("colors", default_colors)
        alpha = kwargs.get("alpha", [1] * len(results_df.controller.unique()))
        linestyles = kwargs.get("linestyles", ["-"] * len(results_df.controller.unique()))
        extra_measurements = copy(extra_measurements)
        for measurement in extra_measurements:
            if measurement not in results_df.measurement.values:
                # logger.warning("Measurement {} not in results dataframe".format(measurement))
                extra_measurements.remove(measurement)
        axs = kwargs.get("axs")
        num_plots = len(self.x_indices) + len(self.u_indices) + len(extra_measurements)

        if axs is None:
            fig, axs = plt.subplots(num_plots, 1, sharex=True)
            fig.set_size_inches(10, 4 * num_plots)
        else:
            assert axs.shape[0] == num_plots
            fig = axs[0].get_figure()

        axs = np.array(axs)  # Also a np.array for num_plots = 1

        num = -1
        for controller in results_df.controller.unique():
            num += 1
            for scenario in results_df.scenario.unique():
                for rollout in results_df.rollout.unique():
                    mask = (
                        (results_df.controller == controller)
                        & (results_df.scenario == scenario)
                        & (results_df.rollout == rollout)
                    )

                    for i, state_label in enumerate(self.x_labels):
                        ax = axs[i]
                        state_mask = mask & (results_df.measurement.values == state_label)
                        unsafe_mask = state_mask & (results_df.unsafe.values == True)
                        safe_mask = state_mask & (results_df.unsafe.values == False)
                        ax.plot(
                            results_df[safe_mask].t,
                            results_df[safe_mask].value,
                            ".",
                            color="green",
                            alpha=alpha[num],
                            # ls=linestyles[num],
                        )
                        ax.plot(
                            results_df[unsafe_mask].t, results_df[unsafe_mask].value, ".", color="red", alpha=alpha[num]
                        )
                        ax.set_ylabel(state_label)

                    for i, control_label in enumerate(self.u_labels):
                        ax = axs[len(self.x_labels) + i]
                        control_mask = mask & (results_df.measurement.values == control_label)
                        ax.plot(
                            results_df[control_mask].t,
                            results_df[control_mask].value,
                            color=colors[num],
                            alpha=alpha[num],
                            ls=linestyles[num],
                        )
                        ax.set_ylabel(control_label)
                        ax.set_ylabel(control_label)

                    for i, extra_label in enumerate(extra_measurements):
                        ax = axs[len(self.x_labels) + len(self.u_labels) + i]
                        extra_mask = mask & (results_df.measurement.values == extra_label)
                        ax.plot(
                            results_df[extra_mask].t,
                            results_df[extra_mask].value,
                            color=colors[num],
                            alpha=alpha[num],
                            ls=linestyles[num],
                        )
                        ax.set_ylabel(extra_label)

        axs[-1].set_xlabel("t")
        axs[-1].set_xlim(min(results_df.t), max(results_df.t))

        fig_handle = ("Rollout (time series)", fig)

        if display_plots:
            plt.show()

        return [fig_handle]


class QuadVerticalDynamics(ControlAffineDynamics):
    STATES = ["Y", "YDOT", "PHI", "PHIDOT"]
    CONTROLS = ["T1", "T2"]
    PERIODIC_DIMS = [2]

    def __init__(self, params, **kwargs):
        self.Cd_v = params["Cd_v"]
        self.g = params["g"]
        self.Cd_phi = params["Cd_phi"]
        self.mass = params["mass"]
        self.length = params["length"]
        self.Iyy = params["Iyy"]
        super().__init__(params, **kwargs)

    def open_loop_dynamics(self, state: np.ndarray, time: float = 0) -> np.ndarray:
        f = np.zeros_like(state)
        f[..., 0] = state[..., 1]
        f[..., 1] = -self.Cd_v / self.mass * state[..., 1] - self.g
        f[..., 2] = state[..., 3]
        f[..., 3] = -self.Cd_phi / self.Iyy * state[..., 3]
        return f

    def control_matrix(self, state: np.ndarray, time: float = 0) -> np.ndarray:
        B = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        B[..., 1, 0] = np.cos(state[..., 2]) / self.mass
        B[..., 1, 1] = np.cos(state[..., 2]) / self.mass
        B[..., 3, 0] = -self.length / self.Iyy
        B[..., 3, 1] = self.length / self.Iyy
        return B

    def disturbance_jacobian(self, state: np.ndarray, time: float = 0) -> np.ndarray:
        return np.repeat(np.zeros_like(state)[..., None], 1, axis=-1)

    def state_jacobian(self, state: np.ndarray, control: np.ndarray, time: float = 0) -> np.ndarray:
        J = np.repeat(np.zeros_like(state)[..., None], state.shape[-1], axis=-1)
        J[..., 0, 1] = 1.0
        J[..., 1, 1] = -self.Cd_v / self.mass
        J[..., 1, 2] = -(control[..., 0] + control[..., 1]) * np.sin(state[..., 2]) / self.mass
        J[..., 2, 3] = 1.0
        J[..., 3, 3] = -self.Cd_phi / self.Iyy
        return J


class QuadVerticalDynamicsInstance:
    def __init__(self):
        gravity: float = 9.81
        mass: float = 2.5
        Cd_v: float = 0.25
        drag_coefficient_phi: float = 0.02255
        length_between_copters: float = 1.0
        moment_of_inertia: float = 1.0

        u_min: float = 0
        u_max: float = 0.75 * mass * gravity
        self.dynamics = QuadVerticalDynamics(
            params={
                "Cd_v": Cd_v,
                "g": gravity,
                "Cd_phi": drag_coefficient_phi,
                "mass": mass,
                "length": length_between_copters,
                "Iyy": moment_of_inertia,
                "dt": 0.02,
            }
        )


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
    dynamics_instance = QuadVerticalDynamicsInstance()
    for i, (key, value) in enumerate(local_hjr_dict.items()):
        patching_result = LocalUpdateResult.load("data/4dim/{}.dill".format(value))
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
