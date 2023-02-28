import warnings

import numpy as np
from jax import numpy as jnp

import hj_reachability
from odp.dynamics.quad4d import Quad4D
from refineNCBF.local_hjr_solver.solve_odp import create_global_solver_odp
from refineNCBF.neural_barrier_kinematic_model_interface.certification import load_uncertified_states, \
    load_certified_states
from refineNCBF.utils.files import generate_unique_filename
from refineNCBF.utils.sets import compute_signed_distance, get_mask_boundary_on_both_sides_by_signed_distance
from refineNCBF.utils.tables import flag_states_on_grid, tabularize_dnn
from scripts.pre_constructed_stuff.quadcopter_cbf import load_cbf_feb24

warnings.simplefilter(action='ignore', category=FutureWarning)


def wip_qv_cbf_global_odp(save_result: bool = False):
    dynamics = Quad4D()

    grid = hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj_reachability.sets.Box(
            # [-1.25, -13.7, -3.65, -7.52],
            # [11.25, 8.3, 3.26, 5.62]
            [0, -8, -np.pi, -10],
            [10, 8, np.pi, 10]
        ),
        shape=(101, 51, 101, 51)
    )

    cbf, standardizer, certified_dict = load_cbf_feb24()
    cbvf = -tabularize_dnn(dnn=cbf, grid=grid, standardizer=standardizer)

    avoid_set = cbvf < 0
    reach_set = jnp.zeros_like(avoid_set, dtype=bool)

    terminal_values = compute_signed_distance(~avoid_set)

    solver = create_global_solver_odp(
        dynamics=dynamics,
        grid=grid,
        periodic_dims=[2],
        avoid_set=avoid_set,
        reach_set=reach_set,
        terminal_values=terminal_values,
        max_iterations=40,
        solver_timestep=-.25,
        verbose=True
    )

    initial_values = terminal_values.copy()

    active_set = (
            get_mask_boundary_on_both_sides_by_signed_distance(~avoid_set, 1)
            & ~
            (flag_states_on_grid(
                cell_centerpoints=load_certified_states(certified_dict, standardizer),
                cell_halfwidths=tuple([0.02551] * grid.ndim),
                grid=grid,
                verbose=True,
                save_array=False
            )
             & ~
             flag_states_on_grid(
                 cell_centerpoints=load_uncertified_states(certified_dict, standardizer),
                 cell_halfwidths=tuple([0.02551] * grid.ndim),
                 grid=grid,
                 verbose=True,
                 save_array=False
             )
             )
    )
    result = solver(active_set=active_set, initial_values=initial_values)

    if save_result:
        result.save(generate_unique_filename('data/local_update_results/wip_qv_cbf_global_odp', 'dill'))

    return result


if __name__ == '__main__':
    wip_qv_cbf_global_odp(save_result=True)
