import os

from refineNCBF.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.utils.files import visuals_data_directory, generate_unique_filename
from refineNCBF.utils.visuals import ArraySlice2D, DimName
import matplotlib.lines as mlines
import numpy as np


def general_replay():
    # result = LocalUpdateResult.load("data/local_update_results/wip_qv_cbf_march_odp_20230228_003139.dill")
    result = LocalUpdateResult.load("data/quad4d/20230307_180336.dill")

    ref_index = ArraySlice2D.from_reference_index(
        reference_index=result.get_middle_index(),
        # reference_index=(0, 15, 0, 26),
        free_dim_1=DimName(0, "y"),
        free_dim_2=DimName(2, "theta"),
    )
    # result.create_gif(
    #     reference_slice=ref_index,
    #     verbose=False,
    #     save_path=os.path.join(
    #         visuals_data_directory,
    #         f'{generate_unique_filename("wip_qv_cbf_global_odp_20230228_025337", "gif")}')
    # )

    # result.render_iteration(iteration=-1, reference_slice=ref_index, verbose=False, save_fig=True)

    # plt.pause(0)


def plot_algorithm():
    result = LocalUpdateResult.load("data/quad4d/20230307_180336.dill")
    result.iterations = result.iterations[::2]  # only plot every other iteration (larger visual effect)
    print(len(result))
    ref_index = ArraySlice2D.from_reference_index(
        # reference_index=result.get_middle_index(),
        reference_index=(0, 18, 0, 25),
        # free_dim_1=DimName(2, "theta"),
        free_dim_1=DimName(2, "theta"),
        free_dim_2=DimName(0, "y"),
    )
    import time

    for itr in range(len(result)):
        print(itr)
        result.plot_algorithm(iteration=itr, reference_slice=ref_index)
        time.sleep(2)
    # result.plot_algorithm(iteration=0, reference_slice=ref_index)


def plot_local_v_global_comparison():
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 20

    fig, ax = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={"width_ratios": [5, 4]})
    ax1 = ax[0, 0]
    ax[0, 1].remove()
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax3 = ax[1, 0]
    ax[1, 1].remove()
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    global_qv_cbf_result = LocalUpdateResult.load("data/quad4d/_20230307_183541.dill")
    march_qv_cbf_result = LocalUpdateResult.load("data/quad4d/20230307_180336.dill")
    ref_index = ArraySlice2D.from_reference_index(
        reference_index=global_qv_cbf_result.get_middle_index(),
        # reference_index=(0, 10, 0, 30),
        # reference_index=(0, 18, 0, 25),
        free_dim_1=DimName(2, "Orientation [rad]"),
        free_dim_2=DimName(0, "Height [m]"),
    )
    march_qv_cbf_result.plot_value_function_comparison(
        reference_slice=ref_index,
        title="Vertical Quadcopter, Final Values Comparison",
        label="Vanilla Reachability",
        iteration=-1,
        comparison_result=global_qv_cbf_result,
        comparison_iteration=-1,
        comparison_label="Boundary March",
        legend=False,
        verbose=False,
        ax=ax2,
    )

    march_qv_cbf_result.plot_value_2d_comparison(
        reference_slice=ref_index,
        title="Vertical Quadcopter, Final Values Comparison 2d",
        label="Vanilla Reachability",
        iteration=-1,
        comparison_result=global_qv_cbf_result,
        comparison_iteration=-1,
        comparison_label="Boundary March",
        legend=False,
        verbose=False,
        ax=ax1,
    )

    ref_index = ArraySlice2D.from_reference_index(
        # reference_index=global_qv_cbf_result.get_middle_index(),
        # reference_index=(0, 10, 0, 30),
        reference_index=(0, 18, 0, 25),
        free_dim_1=DimName(2, "Orientation [rad]"),
        free_dim_2=DimName(0, "Height [m]"),
    )
    march_qv_cbf_result.plot_value_function_comparison(
        reference_slice=ref_index,
        title="Vertical Quadcopter, Final Values Comparison",
        label="Vanilla Reachability",
        iteration=-1,
        comparison_result=global_qv_cbf_result,
        comparison_iteration=-1,
        comparison_label="Boundary March",
        legend=False,
        verbose=False,
        ax=ax4,
    )

    march_qv_cbf_result.plot_value_2d_comparison(
        reference_slice=ref_index,
        title="Vertical Quadcopter, Final Values Comparison 2d",
        label="Vanilla Reachability",
        iteration=-1,
        comparison_result=global_qv_cbf_result,
        comparison_iteration=-1,
        comparison_label="Boundary March",
        legend=False,
        verbose=False,
        ax=ax3,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(visuals_data_directory, f'{generate_unique_filename("4dim_full", "pdf")}'))


def plot_rollouts():
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 20
    import os
    import pandas as pd
    from refineNCBF.utils.files import refineNCBF_dir
    from scripts.validate_barrier.visualize_quad4d import (
        QuadVerticalDynamicsInstance,
        SafevUnsafeStateSpaceExperiment,
    )
    from refine_cbfs.cbf import TabularControlAffineCBF
    import numpy as np

    file_name = "230318_2346"
    # Get directory of this file
    dir_path = os.path.join(refineNCBF_dir, "scripts/validate_barrier")
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
        patching_result = LocalUpdateResult.load("data/quad4d/{}.dill".format(value))
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

    fig, ax = plt.subplots(1, 3, figsize=(16, 10), sharey=True)
    for i, exp in enumerate(["neuralneural", "local_odpneural", "global_odpneural"]):
        if not ("odpneural" in exp or "neuralneural" in exp):
            continue
        df_new = df[df.controller == exp]
        # Remove neural or lqr from end of string
        exp = exp[:-6]

        try:
            resulting_cbf_values = converged_cbvfs[exp]
            result_cbf_init_values = initial_cbfs[exp]
        except KeyError:
            resulting_cbf_values = None
            # resulting_cbf_values = initial_cbfs["local_odp"]
            result_cbf_init_values = initial_cbfs["local_odp"]
        state_space_rollout = SafevUnsafeStateSpaceExperiment("Rollout", start_x=0, x_indices=[1, 0])
        fig_handle = state_space_rollout.plot(
            dynamics_instance.dynamics,
            df_new,
            add_direction=False,
            color=["green"],
            alpha=[0.2],
            ax=ax[i],
        )
        if resulting_cbf_values is not None:
            ax[i].contour(
                x1,
                x2,
                reference_slice.get_sliced_array(resulting_cbf_values).T,
                levels=[0],
                colors="blue",
                linewidths=6,
            )
        # ax.contour(x1, x2, other_slice, levels=[0], colors="k")

        ax[i].contour(
            x1,
            x2,
            reference_slice.get_sliced_array(result_cbf_init_values).T,
            levels=[0],
            colors="k",
            linewidths=6,
            linestyles="dashed",
        )
        # ax.contour(x1, x2, original_slice, levels=[0], colors="orange")
        ax[i].set_xlim(-8.0, 8.0)
        ax[i].set_ylim(0, 10)
        ax[i].set_xlabel("Vertical velocity [m/s]")
    # Remove y label from ax1 and ax2
    ax[0].set_ylabel("Height [m]")
    # ax[1].set_yticks([])
    # ax[2].set_yticks([])
    ax[0].set_title("Neural CBF")
    ax[1].set_title("Patched CBF")
    ax[2].set_title("Global HJR CBF")

    proxies_for_labels = [
        plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", lw=5, alpha=1, linestyle="--"),
        plt.Rectangle((0, 0), 1, 1, fc="w", ec="b", lw=5, alpha=1),
        mlines.Line2D([], [], color="green", alpha=0.5, linewidth=5, linestyle=":"),
        mlines.Line2D([], [], color="red", alpha=0.5, linewidth=5, linestyle=":"),
    ]
    legend_for_labels = [
        r"$\partial \mathcal{C}_{h}$",
        r"$\partial \hat{\mathcal{C}_{h}}$",
        "Safe rollout",
        "Unsafe rollout",
    ]

    minor_x_ticks = np.arange(-8, 8.1, 1)
    minor_y_ticks = np.arange(0, 10.1, 1)
    for a in ax:
        # Set minor ticks to be half of major ticks
        a.tick_params(axis="both", which="minor", labelsize=0)
        a.set_xticks(minor_x_ticks, minor=True)
        # a.set_yticks(minor_y_ticks, minor=True)
        # enable fine grid lines
        a.grid(which="minor", alpha=0.3)
        a.grid(which="major", alpha=1)
    ax[0].set_yticks(minor_y_ticks, minor=True)

    fig.tight_layout()
    lgd = ax[1].legend(
        proxies_for_labels,
        legend_for_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        # fancybox=True,
        # shadow=True,
        ncol=4,
    )
    fig.savefig(
        os.path.join(visuals_data_directory, f'{generate_unique_filename("4dim_rollout", "png")}'),
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
        dpi=400,
    )


def plot_safe_cells_over_time():
    list_of_items = []
    list_of_labels = []
    local_hjr_dict = {
        "global_jax": "_20230307_191036",
        "local_jax": "20230307_025911",
        "local_odp": "20230307_180336",
        "global_odp": "_20230307_183541",
    }
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    for i, (key, value) in enumerate(local_hjr_dict.items()):
        try:
            patching_result = LocalUpdateResult.load("data/local_update_results/{}.dill".format(value))
        except FileNotFoundError:
            try:
                patching_result = LocalUpdateResult.load("data/quad4d/{}.dill".format(value))
            except FileNotFoundError:
                try:
                    patching_result = LocalUpdateResult.load("/data/quad4d/{}.dill".format(value))
                except FileNotFoundError:
                    patching_result = LocalUpdateResult.load("/data_extension/4dim/{}.dill".format(value))
        current_boundary = patching_result.initial_values >= 0
        results = []
        for j, itr in enumerate(patching_result.iterations):
            previous_boundary = current_boundary
            current_boundary = itr.computed_values >= 0
            new_share_unsafe = previous_boundary & ~current_boundary
            res = np.count_nonzero(new_share_unsafe) / np.count_nonzero(previous_boundary)
            results.append(res)
        print("Converged at iteration {}".format(j))
        total_hammies = patching_result.get_total_active_count(j)
        print("Total hammies: {:.2e}".format(total_hammies))
        print("Share safe: {:.2f}".format((np.count_nonzero(current_boundary) / current_boundary.size) * 100))
        print(
            "Initial share safe: {:.2f}".format(
                (np.count_nonzero(patching_result.initial_values >= 0) / patching_result.initial_values.size) * 100
            )
        )

        plt.plot(results, label=key)
    plt.legend()
    plt.savefig(os.path.join(visuals_data_directory, f'{generate_unique_filename("safe_cells_diff", "pdf")}'), dpi=400)


def test():
    global_qv_cbf_result = LocalUpdateResult.load("data/quad4d/_20230307_183541.dill")
    march_qv_cbf_result = LocalUpdateResult.load("data/quad4d/20230307_180336.dill")
    ref_index = ArraySlice2D.from_reference_index(
        reference_index=global_qv_cbf_result.get_middle_index(),
        # reference_index=(0, 10, 0, 30),
        # reference_index=(0, 18, 0, 25),
        free_dim_1=DimName(2, "Orientation [rad]"),
        free_dim_2=DimName(0, "Height [m]"),
    )
    fig, ax = march_qv_cbf_result.plot_value_function_comparison(
        reference_slice=ref_index,
        title="Vertical Quadcopter, Final Values Comparison",
        label="Vanilla Reachability",
        iteration=-1,
        comparison_result=global_qv_cbf_result,
        comparison_iteration=-1,
        comparison_label="Boundary March",
        legend=False,
        verbose=False,
    )
    fig.savefig(
        os.path.join(visuals_data_directory, f'{generate_unique_filename("4dim_final_values", "pdf")}'), dpi=400
    )


if __name__ == "__main__":
    # general_replay()
    plot_algorithm()
    # plot_local_v_global_comparison()
    # plot_rollouts()
    # plot_safe_cells_over_time()
    # test()
