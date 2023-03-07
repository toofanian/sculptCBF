
import os

from refineNCBF.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.utils.files import visuals_data_directory, generate_unique_filename
from refineNCBF.utils.visuals import ArraySlice2D, DimName


def replay():
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

    # result.render_iteration(
    #     iteration=-1,
    #     reference_slice=ref_index,
    #     verbose=False,
    #     save_fig=True
    # )

    compare_result = LocalUpdateResult.load("data/local_update_results/wip_acc_marching_odp_20230227_233418.dill")

    # result.plot_algorithm(iteration=3, reference_slice=ref_index)

    # plt.pause(0)


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


if __name__ == '__main__':
    # replay()
    plot_acc_hammy_comparison()
