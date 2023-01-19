import jax
import numpy as np
from matplotlib import pyplot as plt

from refineNCBF.refining.local_hjr_solver.local_hjr_result import LocalUpdateResult
from refineNCBF.utils.visuals import ArraySlice2D, DimName


def load_result_and_check_visualizations():
    result = LocalUpdateResult.load("data/local_update_results/demo_local_hjr_boundary_decrease_solver_on_quadcopter_vertical_ncbf-20230118_112830.dill")

    print(f'{result.initial_values.size}')
    print(f'{np.count_nonzero(result.initial_values >= 0)}, {np.count_nonzero(result.get_recent_values() >= 0)}')
    print(f'{np.count_nonzero(result.get_total_active_mask())}, {result.initial_values.size}, {np.count_nonzero((result.initial_values <= 1) & (result.initial_values >= -1)) }')

    ref_index = ArraySlice2D.from_reference_index(
        reference_index=(
            jax.numpy.array(9),
            jax.numpy.array(9),
            jax.numpy.array(9),
            jax.numpy.array(9),
        ),
        free_dim_1=DimName(0, 'y'),
        free_dim_2=DimName(2, 'theta')
    )

    result.create_gif(
        reference_slice=ref_index,
        verbose=True
    )

    result.plot_value_function(
        reference_slice=ref_index,
        verbose=True
    )
    #
    # result.plot_value_function_against_truth(
    #     reference_slice=ref_index,
    #     verbose=True
    # )

    plt.pause(0)


if __name__ == '__main__':
    load_result_and_check_visualizations()
