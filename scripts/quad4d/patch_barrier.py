import numpy as np
import hj_reachability
import jax.numpy as jnp

from odp.dynamics.quad4d import Quad4D

from refineNCBF.utils.files import generate_unique_filename
from refineNCBF.local_hjr_solver import SolverAccuracyEnum
from refineNCBF.utils.sets import compute_signed_distance, get_mask_boundary_by_dilation

from refineNCBF.neural_barrier_interface.load_interface import load_cbf
from refineNCBF.neural_barrier_interface.certification import (
    load_certified_states,
    load_uncertified_states,
)
from refineNCBF.utils.tables import flag_states_on_grid, tabularize_dnn

# Add current folder to path
import sys

sys.path.append(".")
from dynamics import quadcopter_vertical_jax_hj
from learned_cbf import quad4d_learned_barrier_params


def main(args):
    if args.solver == "jax":
        dynamics = quadcopter_vertical_jax_hj
    elif args.solver == "odp":
        dynamics = Quad4D()
    else:
        raise ValueError("Unknown solver: {}".format(args.solver))

    grid = hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj_reachability.sets.Box(
            [0, -8, -np.pi, -10],
            [10, 8, np.pi, 10],
        ),
        shape=(75, 41, 75, 41),
        periodic_dims=[2],
    )

    cbf, standardizer, certified_dict = load_cbf(quad4d_learned_barrier_params)
    initial_cbvf = -tabularize_dnn(dnn=cbf, grid=grid, standardizer=standardizer)

    avoid_set = initial_cbvf < 0
    reach_set = jnp.zeros_like(avoid_set, dtype=bool)

    # 2 options for terminal values (\ell(x))
    if args.use_sdf:
        terminal_values = compute_signed_distance(~avoid_set)
    else:
        terminal_values = initial_cbvf * 1e4

    if args.solver == "odp" and args.dp_method == "global":
        from refineNCBF.local_hjr_solver.solve_odp import create_decrease_global_solver_odp

        dp_solver = create_decrease_global_solver_odp

    elif args.solver == "odp" and args.dp_method == "patch":
        from refineNCBF.local_hjr_solver.solve_odp import create_marching_solver_odp

        dp_solver = create_marching_solver_odp

    elif args.solver == "jax" and args.dp_method == "global":
        from refineNCBF.local_hjr_solver.solve import LocalHjrSolver

        dp_solver = LocalHjrSolver.as_global_decrease_solver

    elif args.solver == "jax" and args.dp_method == "patch":
        from refineNCBF.local_hjr_solver.solve import LocalHjrSolver

        dp_solver = LocalHjrSolver.as_marching_solver

    else:
        raise ValueError("Invalid solver and dp_method combination")
    # FIXME: Make it such that solver_accuracy is a parameter of the solver for odp (not integration_scheme)
    solver = dp_solver(
        dynamics=dynamics,
        grid=grid,
        avoid_set=avoid_set,
        reach_set=reach_set,
        terminal_values=terminal_values,
        max_iterations=50,
        solver_timestep=-0.2,
        change_fraction=0.999,
        solver_accuracy=SolverAccuracyEnum.CUSTOMODP,
        hamiltonian_atol=0.01,
        verbose=True,
        solver_global_minimizing=True,
    )

    initial_values = terminal_values.copy()

    if args.dp_method == "global":
        active_set = jnp.ones_like(initial_values, dtype=bool)
    elif args.dp_method == "patch":
        active_set = get_mask_boundary_by_dilation(~avoid_set, 2, 2) & ~(
            flag_states_on_grid(
                cell_centerpoints=load_certified_states(certified_dict, standardizer),
                cell_halfwidths=tuple([quad4d_learned_barrier_params.cellwidths] * grid.ndim),
                grid=grid,
                verbose=True,
                save_array=False,
            )
            & ~flag_states_on_grid(
                cell_centerpoints=load_uncertified_states(certified_dict, standardizer),
                cell_halfwidths=tuple([quad4d_learned_barrier_params.cellwidths] * grid.ndim),
                grid=grid,
                verbose=True,
                save_array=False,
            )
        )
    else:
        raise ValueError("Invalid dp_method")

    # TODO: Solver should take in params and store those too
    result = solver(active_set=active_set, initial_values=initial_values)
    filename = generate_unique_filename("data/quad4d/", "dill")
    result.save(filename)


if __name__ == "__main__":
    # Add parser for command line arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dp_method", type=str, help="global or patch", required=False, default="patch")
    parser.add_argument(
        "--use_sdf",
        type=bool,
        help="Initialize terminal values with signed distance function",
        required=False,
        default=False,
    )
    parser.add_argument("--solver", type=str, help="jax or odp", required=False, default="odp")
    args = parser.parse_args()
    # Print out all the args by iterating through them
    print("Arguments:")
    for arg in vars(args):
        print("\t", arg, "\t", getattr(args, arg))
    main(args)
