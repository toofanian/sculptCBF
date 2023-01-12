import jax
import numpy as np
from matplotlib import pyplot as plt

from refineNCBF.refining.local_hjr_solver.local_hjr_result import LocalUpdateResult
from refineNCBF.utils.visuals import ArraySlice2D


def load_result_and_check_visualizations():
    result = LocalUpdateResult.load("data/local_update_results/demo_local_hjr_boundary_decrease_solver_on_quadcopter_vertical_ncbf.dill")

    print(f'{result.initial_values.size}')
    print(f'{np.count_nonzero(result.initial_values >= 0)}, {np.count_nonzero(result.get_recent_values() >= 0)}')

    ref_index = ArraySlice2D.from_reference_index(
        reference_index=(
            jax.numpy.array(15),
            jax.numpy.array(result.hj_setup.grid.states.shape[1]) // 2,
            jax.numpy.array(15),
            jax.numpy.array(result.hj_setup.grid.states.shape[3]) // 2,
        ),
        free_dim_1=1,
        free_dim_2=3
    )

    result.create_gif(
        reference_slice=ref_index,
        verbose=True
    )

    result.plot_value_function(
        reference_slice=ref_index,
        verbose=True
    )

    plt.pause(0)

if __name__ == '__main__':
    load_result_and_check_visualizations()