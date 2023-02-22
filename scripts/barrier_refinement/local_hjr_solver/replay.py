import os

import jax
import numpy as np
from matplotlib import pyplot as plt

from refineNCBF.refining.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.utils.files import visuals_data_directory, generate_unique_filename, construct_full_path
from refineNCBF.utils.visuals import ArraySlice2D, DimName


def load_result_and_check_visualizations():
    result = LocalUpdateResult.load("data/local_update_results/wip_acc_global_odp_20230222_043457.dill")

    ref_index = ArraySlice2D.from_reference_index(
        reference_index=(1, 0, 0),
        free_dim_1=DimName(1, 'rel_vel'),
        free_dim_2=DimName(2, 'rel_dis')
    )

    result.create_gif(
        reference_slice=ref_index,
        verbose=False,
        save_path=os.path.join(
            visuals_data_directory,
            f'{generate_unique_filename("demo_local_hjr_boundary_decrease_solver_quadcopter_vertical", "gif")}')
    )

    result.plot_value_function(
        iteration=16,
        reference_slice=ref_index,
        verbose=True,
    )

    plt.pause(0)


if __name__ == '__main__':
    load_result_and_check_visualizations()
