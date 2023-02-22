import hj_reachability
import jax
import numpy as np
from matplotlib import pyplot as plt
from odp.Plots import PlotOptions
import odp.Grid

from refineNCBF.dynamic_systems.implementations.quadcopter_fixed_policy import load_quadcopter_sac_jax_hj
from refineNCBF.refining.hj_reachability_interface.hj_step import hj_step
from refineNCBF.refining.hj_reachability_interface.hj_value_postprocessors import ReachAvoid
from refineNCBF.utils.files import construct_full_path, generate_unique_filename, FilePathRelative
from refineNCBF.utils.sets import compute_signed_distance
from refineNCBF.utils.visuals import ArraySlice2D, DimName

# import matplotlib
# matplotlib.use('TkAgg')


def wip_qv_sac_vanilla_jax():
    print('doing jax')
    grid = hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj_reachability.sets.Box(
            [0, -8, -jax.numpy.pi/2, -3],
            [10, 8, jax.numpy.pi/2, 3]
        ),
        shape=(101, 51, 61, 51)
    )

    dynamics = load_quadcopter_sac_jax_hj(grid=grid, relative_path='data/trained_NCBFs/feb18/best_model-3.zip')

    avoid_set = (
            (grid.states[..., 0] < 1)
            |
            (grid.states[..., 0] > 9)
    )

    terminal_values = compute_signed_distance(~avoid_set)
    initial_values = terminal_values.copy()
    running_values = initial_values.copy()

    solver_settings = hj_reachability.SolverSettings(
        value_postprocessor=ReachAvoid.from_array(values=terminal_values),
    )

    for i in range(20):
        next_values = hj_step(
            dynamics=dynamics,
            grid=grid,
            solver_settings=solver_settings,
            initial_values=running_values,
            time_start=0,
            time_target=-.25
        )
        np.save(
            construct_full_path(generate_unique_filename('data/try_fixed_policy'+'_run_midtime_1', 'npy')),
            next_values
        )
        running_values = next_values




def render_result(relative_path: FilePathRelative):
    print('rendering')
    running_values = np.load(construct_full_path(relative_path))

    grid = hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj_reachability.sets.Box(
            [-1, -8, -jax.numpy.pi, -10],
            [11, 8, jax.numpy.pi, 10]
        ),
        shape=(101, 101, 101, 101)
    )

    avoid_set = (
            (grid.states[..., 0] < 0)
            |
            (grid.states[..., 0] > 10)
    )

    terminal_values = compute_signed_distance(~avoid_set)

    reference_slice = ArraySlice2D.from_reference_index(
        reference_index=(10, 10, 10, 10),
        free_dim_1=DimName(0, 'y'),
        free_dim_2=DimName(2, 'theta'),
    )

    x1, x2 = np.meshgrid(
        grid.coordinate_vectors[reference_slice.free_dim_1.dim],
        grid.coordinate_vectors[reference_slice.free_dim_2.dim]
    )

    proxies_for_labels = [
        plt.Rectangle((0, 0), 1, 1, fc='b', ec='w', alpha=.5),
        plt.Rectangle((0, 0), 1, 1, fc='w', ec='b', alpha=1),
    ]

    legend_for_labels = [
        'result',
        'result viability kernel',
    ]

    fig, ax = plt.figure(figsize=(9, 7)), plt.axes(projection='3d')
    ax.set(title='value function')

    ax.plot_surface(
        x1, x2, reference_slice.get_sliced_array(running_values).T,
        cmap='Blues', edgecolor='none', alpha=.5
    )
    ax.contour3D(
        x1, x2, reference_slice.get_sliced_array(running_values).T,
        levels=[0], colors=['b']
    )

    ax.contour3D(
        x1, x2, reference_slice.get_sliced_array(terminal_values).T,
        levels=[0], colors=['k'], linestyles=['--']
    )

    ax.legend(proxies_for_labels, legend_for_labels, loc='upper right')
    ax.set_xlabel(reference_slice.free_dim_1.name)
    ax.set_ylabel(reference_slice.free_dim_2.name)

    plt.show(block=False)
    plt.pause(0)


if __name__ == '__main__':
    # wip_qv_sac_vanilla_jax()
    render_result(relative_path='data/try_fixed_policy_run_bigtime_1_20230221_230229.npy')
