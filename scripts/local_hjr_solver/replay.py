import matplotlib
from matplotlib import pyplot as plt

from refineNCBF.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.utils.visuals import ArraySlice2D, DimName

matplotlib.use('TkAgg')


def load_result_and_check_visualizations():
    result = LocalUpdateResult.load("data/local_update_results/wip_qv_march_odp_20230224_234542.dill")

    # ref_index = ArraySlice2D.from_reference_index(
    #     reference_index=(1, 0, 0),
    #     free_dim_1=DimName(1, 'rel_vel'),
    #     free_dim_2=DimName(2, 'rel_dis')
    # )

    ref_index = ArraySlice2D.from_reference_index(
        reference_index=(25, 12, 25, 12),
        free_dim_1=DimName(0, 'y'),
        free_dim_2=DimName(2, 'theta')
    )
    #
    # result.create_gif(
    #     reference_slice=ref_index,
    #     verbose=False,
    #     save_path=os.path.join(
    #         visuals_data_directory,
    #         f'{generate_unique_filename("demo_local_hjr_boundary_decrease_solver_quadcopter_vertical", "gif")}')
    # )

    result.render_iteration(
        iteration=-1,
        reference_slice=ref_index,
        verbose=True,
    )

    plt.pause(0)


if __name__ == '__main__':
    load_result_and_check_visualizations()
