import warnings
from typing import List, Optional

import attr
import dill
import hj_reachability
import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt, animation

from refineNCBF.refining.hj_reachability_interface.hj_setup import HjSetup
from refineNCBF.refining.hj_reachability_interface.hj_step import hj_step
from refineNCBF.refining.hj_reachability_interface.hj_value_postprocessors import ReachAvoid
from refineNCBF.utils.files import FilePathRelative, check_if_file_exists, construct_full_path
from refineNCBF.utils.sets import compute_signed_distance
from refineNCBF.utils.types import MaskNd, ArrayNd
from refineNCBF.utils.visuals import ArraySlice2D

import matplotlib
matplotlib.use('TkAgg')

warnings.simplefilter(action='ignore', category=FutureWarning)


@attr.dataclass
class LocalUpdateResultIteration:
    active_set_pre_filtered: MaskNd
    active_set_expanded: MaskNd
    computed_values: ArrayNd
    active_set_post_filtered: MaskNd
    solver_info: Optional[List[List[int]]] = None

    @classmethod
    def from_parts(
            cls,
            active_set_pre_filtered: MaskNd,
            active_set_expanded: MaskNd,
            values_next: ArrayNd,
            active_set_post_filtered: MaskNd,
            solver_info: Optional[List[List[int]]] = None,
    ):
        return cls(
            active_set_pre_filtered=active_set_pre_filtered,
            active_set_expanded=active_set_expanded,
            computed_values=values_next,
            active_set_post_filtered=active_set_post_filtered,
            solver_info=solver_info,
        )


@attr.dataclass
class LocalUpdateResult:
    hj_setup: HjSetup
    avoid_set: MaskNd
    reach_set: MaskNd
    seed_set: MaskNd
    initial_values: ArrayNd
    iterations: List[LocalUpdateResultIteration] = attr.ib(factory=list)

    @classmethod
    def from_parts(
            cls,
            hj_setup: HjSetup,
            avoid_set: MaskNd,
            seed_set: MaskNd,
            initial_values: ArrayNd,
            reach_set: Optional[MaskNd] = None
    ):
        if reach_set is None:
            reach_set = jnp.zeros_like(avoid_set, dtype=bool)

        return cls(hj_setup, avoid_set, reach_set, seed_set, initial_values)

    def __len__(self):
        return len(self.iterations)

    def add_iteration(self, iteration: LocalUpdateResultIteration):
        self.iterations.append(iteration)

    def save(self, file_path: FilePathRelative):
        full_path = construct_full_path(file_path)
        check_if_file_exists(full_path)
        with open(full_path, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, file_path: FilePathRelative) -> "LocalUpdateResult":
        full_path = construct_full_path(file_path)
        with open(full_path, "rb") as f:
            cls = dill.load(f)
        return cls

    def get_recent_set_input(self) -> MaskNd:
        return self.seed_set if len(self.iterations) == 0 else self.iterations[-1].active_set_post_filtered

    def get_recent_set_for_compute(self) -> MaskNd:
        if len(self.iterations) == 0:
            raise ValueError("No iterations have been computed yet.")
        else:
            return self.iterations[-1].active_set_expanded

    def get_total_active_count(self, up_to_iteration: int) -> int:
        return sum([
            jnp.count_nonzero(iteration.active_set_expanded)
            for iteration
            in self.iterations[:up_to_iteration]
        ])

    def get_total_active_mask(self) -> MaskNd:
        total_active_mask = jnp.zeros_like(self.seed_set, dtype=bool)
        for iteration in self.iterations:
            total_active_mask = total_active_mask | iteration.active_set_expanded
        return total_active_mask

    def get_recent_values(self) -> ArrayNd:
        return self.initial_values if len(self) == 0 else self.iterations[-1].computed_values

    def get_recent_values_list(self, span: int = 1) -> List[ArrayNd]:
        recent_values_list = []
        for i in range(span):
            if len(self.iterations) > i:
                recent_values_list.append(self.iterations[-1 - i].computed_values)
            else:
                recent_values_list.append(self.initial_values)
                break
        return recent_values_list

    def get_viability_kernel(self) -> MaskNd:
        return self.get_recent_values() >= 0

    def create_gif(
            self,
            reference_slice: ArraySlice2D,
            verbose: bool = True,
            save_path: Optional[FilePathRelative] = None
    ):
        proxies_for_labels = [
            plt.Rectangle((0, 0), 1, 1, fc='r', ec='w', alpha=.3),
            plt.Rectangle((0, 0), 1, 1, fc='g', ec='w', alpha=.3),
            plt.Rectangle((0, 0), 1, 1, fc='k', ec='w', alpha=.3),
            plt.Rectangle((0, 0), 1, 1, fc='w', ec='k', alpha=.7),
            plt.Rectangle((0, 0), 1, 1, fc='b', ec='w', alpha=.2),
            plt.Rectangle((0, 0), 1, 1, fc='b', ec='w', alpha=.4),
            plt.Rectangle((0, 0), 1, 1, fc='w', ec='k', linestyle='--'),
            plt.Rectangle((0, 0), 1, 1, fc='w', ec='b', linestyle='--'),

        ]
        legend_for_labels = [
            'avoid set',
            'reach set',
            'seed set',
            'initial zero levelset',
            'active set',
            'changed set',
            'running viability kernel',
            'computed subgrid'
        ]

        def animate(i):
            ax.clear()

            ax.set(title=f"iteration: {i}, total active: {self.get_total_active_count(i)} of {self.avoid_set.size} \nSliced at {reference_slice.slice_string}")
            ax.set_xlabel(reference_slice.free_dim_1.name)
            ax.set_ylabel(reference_slice.free_dim_2.name)

            # always have reach and avoid set up
            ax.contourf(
                self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.avoid_set).T,
                levels=[0, .5], colors=['r'], alpha=.3
            )

            ax.contourf(
                self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.reach_set).T,
                levels=[0, .5], colors=['g'], alpha=.3
            )

            # always have seed set up
            ax.contourf(
                self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.seed_set).T,
                levels=[0, .5], colors=['k'], alpha=.3
            )

            # always have initial zero levelset up
            ax.contour(
                self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                reference_slice.get_sliced_array(self.initial_values).T,
                levels=[0], colors=['k'], alpha=.7
            )

            if self.iterations[i].solver_info is not None:
                ax.axvline(
                    x=self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_1.dim][
                          self.iterations[i].solver_info[0][reference_slice.free_dim_1.dim]] - 1,
                    color='b', linestyle='--'
                )
                ax.axvline(
                    x=self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_1.dim][
                        self.iterations[i].solver_info[1][reference_slice.free_dim_1.dim]],
                    color='b', linestyle='--'
                )
                ax.axhline(
                    y=self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_2.dim][
                          self.iterations[i].solver_info[0][reference_slice.free_dim_2.dim]] - 1,
                    color='b', linestyle='--'
                )
                ax.axhline(
                    y=self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_2.dim][
                        self.iterations[i].solver_info[1][reference_slice.free_dim_2.dim]],
                    color='b', linestyle='--'
                )

            ax.contourf(
                self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.iterations[i].active_set_expanded).T,
                levels=[0, .5], colors=['b'], alpha=.2
            )

            ax.contourf(
                self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.iterations[i].active_set_post_filtered).T,
                levels=[0, 0.5], colors=['b'], alpha=.2
            )

            values = self.initial_values if i == 0 else self.iterations[i - 1].computed_values
            ax.contour(
                self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                reference_slice.get_sliced_array(values).T,
                levels=[0], colors=['k'], linestyles=['--']
            )

            ax.legend(proxies_for_labels, legend_for_labels, loc='upper right')

            return ax

        fig, ax = plt.subplots(figsize=(9, 7))
        anim = animation.FuncAnimation(fig, animate, frames=len(self), interval=100)

        if save_path is not None:
            if not check_if_file_exists(save_path):
                anim.save(construct_full_path(save_path), writer='imagemagick', fps=4)
            else:
                print(f"file {save_path} already exists, not saving animation")

        if verbose:
            plt.show(block=False)

    def plot_value_function_against_truth(
            self,
            reference_slice: ArraySlice2D,
            target_time: float = -10,
            verbose: bool = False,
            save_path: Optional[FilePathRelative] = None
    ):
        if save_path is not None:
            raise NotImplementedError('saving not implemented yet')

        final_values = self.get_recent_values()

        terminal_values = compute_signed_distance(~self.avoid_set)
        solver_settings = hj_reachability.solver.SolverSettings.with_accuracy(
            accuracy=hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH,
            value_postprocessor=ReachAvoid.from_array(terminal_values, self.reach_set)
        )

        truth = hj_step(
            hj_setup=self.hj_setup,
            solver_settings=solver_settings,
            initial_values=terminal_values,
            time_start=0.,
            time_target=target_time,
            progress_bar=True
        )

        total_active_mask = self.get_total_active_mask()

        x1, x2 = np.meshgrid(
            self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_2.dim]
        )

        proxies_for_labels = [
            plt.Rectangle((0, 0), 1, 1, fc='k', ec='w', alpha=.3),
            plt.Rectangle((0, 0), 1, 1, fc='w', ec='k', alpha=1),
            plt.Rectangle((0, 0), 1, 1, fc='b', ec='w', alpha=.5),
            plt.Rectangle((0, 0), 1, 1, fc='w', ec='b', alpha=1),
            plt.Rectangle((0, 0), 1, 1, fc='r', ec='w', alpha=.5),
            plt.Rectangle((0, 0), 1, 1, fc='w', ec='r', alpha=1),
            plt.Rectangle((0, 0), 1, 1, fc='y', ec='y', alpha=.2),
            plt.Rectangle((0, 0), 1, 1, fc='g', ec='g', alpha=.2),
        ]

        legend_for_labels = [
            'initial'
            'initial kernel'
            'result',
            'result kernel',
            'truth',
            'truth kernel',
            'inaccurate after compute',
            'accurate after compute'
        ]

        fig, ax = plt.figure(figsize=(9, 7)), plt.axes(projection='3d')
        ax.set(title='value function against truth')

        ax.plot_surface(
            x1, x2, reference_slice.get_sliced_array(final_values).T,
            cmap='Blues', edgecolor='none', alpha=.5
        )
        ax.contour3D(
            x1, x2, reference_slice.get_sliced_array(final_values).T,
            levels=[0], colors=['b']
        )

        ax.plot_surface(
            x1, x2, reference_slice.get_sliced_array(truth).T,
            cmap='Reds', edgecolor='none', alpha=.5
        )
        ax.contour3D(
            x1, x2, reference_slice.get_sliced_array(truth).T,
            levels=[0], colors=['r'], linestyles=['--']
        )

        ax.plot_surface(
            x1, x2, reference_slice.get_sliced_array(self.initial_values).T,
            cmap='Greys', edgecolor='none', alpha=.3
        )
        ax.contour3D(
            x1, x2, reference_slice.get_sliced_array(self.initial_values).T,
            levels=[0], colors=['k'],
        )

        ax.contourf3D(
            x1, x2, ~reference_slice.get_sliced_array(~jnp.isclose(truth, final_values, atol=.5) & total_active_mask).T,
            levels=[0, .01], colors=['y'], alpha=.2
        )

        ax.contour3D(
            x1, x2, ~reference_slice.get_sliced_array(~jnp.isclose(truth, final_values, atol=.5) & total_active_mask).T,
            levels=[0], colors=['y'],
        )
        ax.contourf3D(
            x1, x2, ~reference_slice.get_sliced_array(jnp.isclose(truth, final_values, atol=.5) & total_active_mask).T,
            levels=[0, .01], colors=['g'], alpha=.2
        )
        ax.contour3D(
            x1, x2, ~reference_slice.get_sliced_array(jnp.isclose(truth, final_values, atol=.5) & total_active_mask).T,
            levels=[0], colors=['g'], alpha=1
        )
        ax.legend(proxies_for_labels, legend_for_labels, loc='upper right')
        ax.set_xlabel(reference_slice.free_dim_1.name)
        ax.set_ylabel(reference_slice.free_dim_2.name)
        print(f'total inaccurate states: {np.count_nonzero(~jnp.isclose(truth, final_values, atol=.5) & total_active_mask)}')

        if verbose:
            plt.show()

        return fig, ax

    def plot_value_function(
            self,
            reference_slice: ArraySlice2D,
            verbose: bool = False,
            save_path: Optional[FilePathRelative] = None
    ):
        if save_path is not None:
            raise NotImplementedError('saving not implemented yet')

        final_values = self.get_recent_values()

        x1, x2 = np.meshgrid(
            self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.hj_setup.grid.coordinate_vectors[reference_slice.free_dim_2.dim]
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
        ax.set(title='value function against truth')

        ax.plot_surface(
            x1, x2, reference_slice.get_sliced_array(final_values).T,
            cmap='Blues', edgecolor='none', alpha=.5
        )
        ax.contour3D(
            x1, x2, reference_slice.get_sliced_array(final_values).T,
            levels=[0], colors=['b']
        )

        ax.contour3D(
            x1, x2, reference_slice.get_sliced_array(self.initial_values).T,
            levels=[0], colors=['k'], linestyles=['--']
        )

        ax.legend(proxies_for_labels, legend_for_labels, loc='upper right')
        ax.set_xlabel(reference_slice.free_dim_1.name)
        ax.set_ylabel(reference_slice.free_dim_2.name)

        if verbose:
            plt.show(block=False)

        return fig, ax
