
import os

from refineNCBF.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.utils.files import visuals_data_directory, generate_unique_filename
from refineNCBF.utils.visuals import ArraySlice2D, DimName


def general_replay():
    # result = LocalUpdateResult.load("data/local_update_results/wip_qv_cbf_march_odp_20230228_003139.dill")
    result = LocalUpdateResult.load("data/local_update_results/result_acc_global_odp_3x75x75.dill")

    ref_index = ArraySlice2D.from_reference_index(
        reference_index=(1, 0, 0),
        free_dim_1=DimName(1, 'rel_vel'),
        free_dim_2=DimName(2, 'rel_dis')
    )
    
    # ref_index = ArraySlice2D.from_reference_index(
    #     # reference_index=result.get_middle_index(),
    #     reference_index=(0, 15, 0, 26),
    #     free_dim_1=DimName(0, 'y'),
    #     free_dim_2=DimName(2, 'theta')
    # )
    # result.create_gif(
    #     reference_slice=ref_index,
    #     verbose=False,
    #     save_path=os.path.join(
    #         visuals_data_directory,
    #         f'{generate_unique_filename("wip_qv_cbf_global_odp_20230228_025337", "gif")}')
    # )

    result.render_iteration(
        iteration=-1,
        reference_slice=ref_index,
        verbose=False,
        save_fig=True
    )

    # plt.pause(0)


def plot_algorithm():
    march_acc_result = LocalUpdateResult.load("data/local_update_results/result_acc_march_jax_3x75x75.dill")
    ref_index = ArraySlice2D.from_reference_index(
        reference_index=(1, 0, 0),
        free_dim_1=DimName(1, 'Relative Velocity'),
        free_dim_2=DimName(2, 'Relative Distance')
    )
    march_acc_result.plot_algorithm(iteration=3, reference_slice=ref_index)


def plot_acc_hammy_comparison():
    global_acc_result = LocalUpdateResult.load("data/local_update_results/result_acc_global_odp_3x75x75.dill")
    march_acc_result = LocalUpdateResult.load("data/local_update_results/result_acc_march_odp_3x75x75.dill")

    global_acc_result.plot_kernel_accuracy_vs_hammys(
        title='Active Cruise Control Kernel Compute Costs',
        label='Vanilla Reachability',
        compare_results=[march_acc_result],
        compare_labels=['Boundary March'],
        x_scale='log',
        y_scale='log',
        ignore_dim=(0,)
    )


def plot_qv_cbf_hammy_comparison():
    global_acc_result = LocalUpdateResult.load("data/local_update_results/result_qv_cbf_global_odp_75x41x75x41.dill")
    march_acc_result = LocalUpdateResult.load("data/local_update_results/result_qv_cbf_march_odp_75x41x75x41.dill")

    global_acc_result.plot_kernel_accuracy_vs_hammys(
        title='Quadcopter Vertical CBF Warmstart Kernel Compute Costs',
        label='Vanilla Reachability',
        compare_results=[march_acc_result],
        compare_labels=['Boundary March'],
        x_scale='log',
        y_scale='log',
        ignore_dim=(0,)
    )


def plot_acc_initial():
    result = LocalUpdateResult.load("data/local_update_results/result_acc_march_odp_3x75x75.dill")
    ref_index = ArraySlice2D.from_reference_index(
        reference_index=(1, 0, 0),
        free_dim_1=DimName(1, 'Relative Velocity [m/s]'),
        free_dim_2=DimName(2, 'Relative Distance [m]')
    )
    result.plot_failure_set(ref_index)
    result.plot_initial_values(ref_index, verbose=False, save_fig=True)


def plot_qv_cbf_initial():
    result = LocalUpdateResult.load("data/local_update_results/result_qv_cbf_global_odp_75x41x75x41.dill")
    ref_index = ArraySlice2D.from_reference_index(
        reference_index=result.get_middle_index(),
        free_dim_1=DimName(0, 'Height [m]'),
        free_dim_2=DimName(2, 'Orientation [rad]')
    )
    result.plot_failure_set(ref_index)
    result.plot_initial_values(ref_index, verbose=False, save_fig=True)


def plot_acc_final():
    global_acc_result = LocalUpdateResult.load("data/local_update_results/result_acc_global_odp_3x75x75.dill")
    march_acc_result = LocalUpdateResult.load("data/local_update_results/result_acc_march_jax_3x75x75.dill")
    ref_index = ArraySlice2D.from_reference_index(
        reference_index=(1, 0, 0),
        free_dim_1=DimName(1, 'Relative Velocity'),
        free_dim_2=DimName(2, 'Relative Distance')
    )
    global_acc_result.plot_value_function_comparison(
        reference_slice=ref_index,
        title='Active Cruise Control, Final Values Comparison',
        label='Vanilla Reachability',
        iteration=-1,
        comparison_result=march_acc_result,
        comparison_iteration=-1,
        comparison_label='Boundary March',
        verbose=False,
        save_fig=True
    )


def plot_acc_march_iterations():
    march_acc_result = LocalUpdateResult.load("data/local_update_results/result_acc_march_jax_3x75x75.dill")
    ref_index = ArraySlice2D.from_reference_index(
        reference_index=(1, 0, 0),
        free_dim_1=DimName(1, 'Relative Velocity'),
        free_dim_2=DimName(2, 'Relative Distance')
    )
    march_acc_result.render_iteration(
        iteration=0,
        reference_slice=ref_index,
        legend=False,
        verbose=False,
        save_fig=True
    )
    march_acc_result.render_iteration(
        iteration=1,
        reference_slice=ref_index,
        legend=False,
        verbose=False,
        save_fig=True
    )
    march_acc_result.render_iteration(
        iteration=10,
        reference_slice=ref_index,
        legend=False,
        verbose=False,
        save_fig=True
    )
    march_acc_result.render_iteration(
        iteration=20,
        reference_slice=ref_index,
        legend=False,
        verbose=False,
        save_fig=True
    )
    march_acc_result.render_iteration(
        iteration=-1,
        reference_slice=ref_index,
        legend=False,
        verbose=False,
        save_fig=True
    )


def plot_acc_global_iterations():
    global_acc_result = LocalUpdateResult.load("data/local_update_results/result_acc_global_jax_3x75x75.dill")
    ref_index = ArraySlice2D.from_reference_index(
        reference_index=(1, 0, 0),
        free_dim_1=DimName(1, 'Relative Velocity'),
        free_dim_2=DimName(2, 'Relative Distance')
    )
    global_acc_result.render_iteration(
        iteration=0,
        reference_slice=ref_index,
        legend=False,
        verbose=False,
        save_fig=True
    )
    global_acc_result.render_iteration(
        iteration=1,
        reference_slice=ref_index,
        legend=False,
        verbose=False,
        save_fig=True
    )
    global_acc_result.render_iteration(
        iteration=20,
        reference_slice=ref_index,
        legend=False,
        verbose=False,
        save_fig=True
    )
    global_acc_result.render_iteration(
        iteration=40,
        reference_slice=ref_index,
        legend=False,
        verbose=False,
        save_fig=True
    )
    global_acc_result.render_iteration(
        iteration=-1,
        reference_slice=ref_index,
        legend=False,
        verbose=False,
        save_fig=True
    )


if __name__ == '__main__':
    # replay()
    # plot_acc_hammy_comparison()
    # plot_qv_cbf_hammy_comparison()
    # plot_acc_initial()
    # plot_acc_final()
    # plot_algorithm()
    # plot_acc_march_iterations()
    plot_qv_cbf_initial()
    # plot_acc_global_iterations()
