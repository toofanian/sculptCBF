import os

from refineNCBF.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.utils.files import visuals_data_directory, generate_unique_filename
from refineNCBF.utils.visuals import ArraySlice2D, DimName


def replay():
    result = LocalUpdateResult.load("data/local_update_results/wip_qv_march_odp_20230227_234704.dill")
    # ref_index = ArraySlice2D.from_reference_index(
    #     reference_index=(1, 0, 0),
    #     free_dim_1=DimName(1, 'rel_vel'),
    #     free_dim_2=DimName(2, 'rel_dis')
    # )
    ref_index = ArraySlice2D.from_reference_index(
        reference_index=result.get_middle_index(),
        free_dim_1=DimName(0, 'y'),
        free_dim_2=DimName(2, 'theta')
    )
    result.create_gif(
        reference_slice=ref_index,
        verbose=False,
        save_path=os.path.join(
            visuals_data_directory,
            f'{generate_unique_filename("wip_qv_march_odp_20230227_234704", "gif")}')
    )
    #
    # result.render_iteration(
    #     iteration=-1,
    #     reference_slice=ref_index,
    #     verbose=True,
    # )

    # plt.pause(0)


if __name__ == '__main__':
    replay()
