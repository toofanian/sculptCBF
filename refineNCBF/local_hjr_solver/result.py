import warnings
from typing import List, Optional, Union

import attr
import dill
import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt, animation

import hj_reachability
from refineNCBF.hj_reachability_interface.hj_step import hj_step
from refineNCBF.hj_reachability_interface.hj_value_postprocessors import ReachAvoid
from refineNCBF.optimized_dp_interface.odp_dynamics import OdpDynamics
from refineNCBF.utils.files import FilePathRelative, check_if_file_exists, construct_refine_ncbf_path, \
    generate_unique_filename
from refineNCBF.utils.types import MaskNd, ArrayNd
from refineNCBF.utils.visuals import ArraySlice2D, ArraySlice1D

warnings.simplefilter(action='ignore', category=FutureWarning)


@attr.dataclass
class LocalUpdateResultIteration:
    active_set_pre_filtered: MaskNd
    active_set_expanded: MaskNd
    computed_values: ArrayNd
    active_set_post_filtered: MaskNd

    @classmethod
    def from_parts(
            cls,
            active_set_pre_filtered: MaskNd,
            active_set_expanded: MaskNd,
            values_next: ArrayNd,
            active_set_post_filtered: MaskNd,
    ):
        return cls(
            active_set_pre_filtered=active_set_pre_filtered,
            active_set_expanded=active_set_expanded,
            computed_values=values_next,
            active_set_post_filtered=active_set_post_filtered,
        )


@attr.dataclass
class LocalUpdateResult:
    local_solver: "LocalHjrSolver"
    dynamics: hj_reachability.Dynamics
    grid: hj_reachability.Grid
    avoid_set: MaskNd
    reach_set: MaskNd
    initial_values: ArrayNd
    terminal_values: ArrayNd
    seed_set: MaskNd

    iterations: List[LocalUpdateResultIteration] = attr.ib(factory=list)

    @classmethod
    def from_parts(
            cls,
            local_solver: "LocalHjrSolver",
            dynamics: Union[hj_reachability.Dynamics, OdpDynamics],
            grid: hj_reachability.Grid,
            avoid_set: MaskNd,
            seed_set: MaskNd,
            initial_values: ArrayNd,
            terminal_values: ArrayNd,
            reach_set: Optional[MaskNd] = None
    ):
        if reach_set is None:
            reach_set = jnp.zeros_like(avoid_set, dtype=bool)

        return cls(
            local_solver=local_solver,
            dynamics=dynamics,
            grid=grid,
            avoid_set=avoid_set,
            reach_set=reach_set,
            initial_values=initial_values,
            terminal_values=terminal_values,
            seed_set=seed_set,
        )

    def __len__(self):
        return len(self.iterations)

    def add_iteration(self, iteration: LocalUpdateResultIteration):
        self.iterations.append(iteration)

    def save(self, file_path: FilePathRelative):
        full_path = construct_refine_ncbf_path(file_path)
        check_if_file_exists(full_path)
        with open(full_path, "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def load(file_path: FilePathRelative) -> "LocalUpdateResult":
        full_path = construct_refine_ncbf_path(file_path)
        with open(full_path, "rb") as f:
            cls = dill.load(f)
        return cls

    def max_diff(self):
        max_diff = jnp.max(jnp.abs(self.get_recent_values() - self.get_previous_values()))
        return max_diff

    def get_previous_values(self):
        if len(self) > 1:
            return self.iterations[-2].computed_values
        else:
            return self.initial_values

    def get_pending_seed_set(self) -> MaskNd:
        if len(self.iterations) == 0:
            return self.seed_set
        else:
            return self.iterations[-1].active_set_post_filtered

    def get_recent_set_for_compute(self) -> MaskNd:
        if len(self) == 0:
            return jnp.ones_like(self.seed_set, dtype=bool)
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

    def plot_value_1d(self, ref_index: ArraySlice1D):
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot(self.grid.coordinate_vectors[ref_index.free_dim_1.dim],
                ref_index.get_sliced_array(self.initial_values),
                )
        for iteration in self.iterations:
            ax.plot(
                self.grid.coordinate_vectors[ref_index.free_dim_1.dim],
                ref_index.get_sliced_array(iteration.computed_values)
            )

        ax.set_xlabel(f'{ref_index.free_dim_1.name}')
        ax.set_ylabel(f'value')

        plt.show(block=False)

    def create_gif(
            self,
            reference_slice: Union[ArraySlice2D, ArraySlice1D],
            verbose: bool = True,
            save_path: Optional[FilePathRelative] = None
    ):
        if isinstance(reference_slice, ArraySlice1D):
            reference_slice = ArraySlice2D.from_array_slice_1d(reference_slice)

        proxies_for_labels = [
            plt.Rectangle((0, 0), 1, 1, fc='r', ec='w', alpha=.3),
            plt.Rectangle((0, 0), 1, 1, fc='g', ec='w', alpha=.3),
            plt.Rectangle((0, 0), 1, 1, fc='k', ec='w', alpha=.3),
            plt.Rectangle((0, 0), 1, 1, fc='w', ec='k', alpha=.7),
            plt.Rectangle((0, 0), 1, 1, fc='b', ec='w', alpha=.2),
            plt.Rectangle((0, 0), 1, 1, fc='b', ec='w', alpha=.4),
            plt.Rectangle((0, 0), 1, 1, fc='w', ec='k', linestyle='--'),

        ]
        legend_for_labels = [
            'avoid set',
            'reach set',
            'seed set',
            'initial zero levelset',
            'active set',
            'changed set',
            'running viability kernel',
        ]

        def render_iteration(i: int):
            ax.clear()

            ax.set(
                title=f"iteration: {i}, total active: {self.get_total_active_count(i)} of {self.avoid_set.size} \nSliced at {reference_slice.slice_string}")
            ax.set_xlabel(reference_slice.free_dim_1.name)
            ax.set_ylabel(reference_slice.free_dim_2.name)

            # always have reach and avoid set up
            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.avoid_set).T,
                levels=[0, .5], colors=['r'], alpha=.3
            )

            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.reach_set).T,
                levels=[0, .5], colors=['g'], alpha=.3
            )

            # always have seed set up
            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.seed_set).T,
                levels=[0, .5], colors=['k'], alpha=.3
            )

            # always have initial zero levelset up
            ax.contour(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                reference_slice.get_sliced_array(self.initial_values).T,
                levels=[0], colors=['k'], alpha=.7
            )

            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.iterations[i].active_set_expanded).T,
                levels=[0, .5], colors=['b'], alpha=.2
            )

            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.iterations[i].active_set_post_filtered).T,
                levels=[0, 0.5], colors=['b'], alpha=.2
            )

            values = self.initial_values if i == 0 else self.iterations[i - 1].computed_values
            ax.contour(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                reference_slice.get_sliced_array(values).T,
                levels=[0], colors=['k'], linestyles=['--']
            )

            ax.legend(proxies_for_labels, legend_for_labels, loc='upper right')

            return ax

        fig, ax = plt.subplots(figsize=(9, 7))
        anim = animation.FuncAnimation(fig, render_iteration, frames=len(self), interval=100)

        if save_path is not None:
            if not check_if_file_exists(save_path):
                anim.save(construct_refine_ncbf_path(save_path), writer='imagemagick', fps=4)
            else:
                print(f"file {save_path} already exists, not saving animation")

        if verbose:
            plt.show(block=False)

    def create_gif_3d(
            self,
            reference_slice: Union[ArraySlice2D, ArraySlice1D],
            verbose: bool = True,
            save_path: Optional[FilePathRelative] = None
    ):
        if isinstance(reference_slice, ArraySlice1D):
            reference_slice = ArraySlice2D.from_array_slice_1d(reference_slice)

        proxies_for_labels = [
            plt.Rectangle((0, 0), 1, 1, fc='b', ec='w', alpha=.5),
            plt.Rectangle((0, 0), 1, 1, fc='w', ec='b', alpha=1),
        ]

        legend_for_labels = [
            'result',
            'result viability kernel',
        ]

        fig, ax = plt.figure(figsize=(9, 7)), plt.axes(projection='3d')

        def render_iteration(i: int):
            values = self.iterations[i].computed_values

            x1, x2 = np.meshgrid(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim]
            )
            ax.set(title='value function')

            ax.plot_surface(
                x1, x2, reference_slice.get_sliced_array(values).T,
                cmap='Blues', edgecolor='none', alpha=.5
            )
            ax.contour3D(
                x1, x2, reference_slice.get_sliced_array(values).T,
                levels=[0], colors=['b']
            )

            ax.contour3D(
                x1, x2, reference_slice.get_sliced_array(self.initial_values).T,
                levels=[0], colors=['k'], linestyles=['--']
            )

            ax.legend(proxies_for_labels, legend_for_labels, loc='upper right')
            ax.set_xlabel(reference_slice.free_dim_1.name)
            ax.set_ylabel(reference_slice.free_dim_2.name)

            return ax

        fig, ax = plt.subplots(figsize=(9, 7))
        anim = animation.FuncAnimation(fig, render_iteration, frames=len(self), interval=100)

        if save_path is not None:
            if not check_if_file_exists(save_path):
                anim.save(construct_refine_ncbf_path(save_path), writer='imagemagick', fps=4)
            else:
                print(f"file {save_path} already exists, not saving animation")

        if verbose:
            plt.show(block=False)

    def render_iteration(self, iteration: int, reference_slice: ArraySlice2D, verbose: bool = True,
                         save_path: str = None):
        fig, ax = plt.subplots(figsize=(9, 7))

        proxies_for_labels = [
            plt.Rectangle((0, 0), 1, 1, fc='r', ec='w', alpha=.3),
            plt.Rectangle((0, 0), 1, 1, fc='g', ec='w', alpha=.3),
            plt.Rectangle((0, 0), 1, 1, fc='k', ec='w', alpha=.3),
            plt.Rectangle((0, 0), 1, 1, fc='w', ec='k', alpha=.7),
            plt.Rectangle((0, 0), 1, 1, fc='b', ec='w', alpha=.2),
            plt.Rectangle((0, 0), 1, 1, fc='b', ec='w', alpha=.4),
            plt.Rectangle((0, 0), 1, 1, fc='w', ec='k', linestyle='--'),
        ]

        legend_for_labels = [
            'avoid set',
            'reach set',
            'seed set',
            'initial zero levelset',
            'active set',
            'changed set',
            'running viability kernel',
        ]

        ax.set(title=f"iteration: {iteration} of {len(self)}")

        ax.set_xlabel(reference_slice.free_dim_1.name)
        ax.set_ylabel(reference_slice.free_dim_2.name)

        # always have reach and avoid set up
        ax.contourf(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            ~reference_slice.get_sliced_array(self.avoid_set).T,
            levels=[0, .5], colors=['r'], alpha=.3
        )

        ax.contourf(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            ~reference_slice.get_sliced_array(self.reach_set).T,
            levels=[0, .5], colors=['g'], alpha=.3
        )

        # always have seed set up
        ax.contourf(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            ~reference_slice.get_sliced_array(self.seed_set).T,
            levels=[0, .5], colors=['k'], alpha=.3
        )

        # always have initial zero levelset up
        ax.contour(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            reference_slice.get_sliced_array(self.initial_values).T,
            levels=[0], colors=['k'], alpha=.7
        )

        ax.contourf(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            ~reference_slice.get_sliced_array(self.iterations[iteration].active_set_expanded).T,
            levels=[0, .5], colors=['b'], alpha=.2
        )

        ax.contourf(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            ~reference_slice.get_sliced_array(self.iterations[iteration].active_set_post_filtered).T,
            levels=[0, 0.5], colors=['b'], alpha=.2
        )

        values = self.initial_values if iteration == 0 else self.iterations[iteration - 1].computed_values
        ax.contour(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            reference_slice.get_sliced_array(values).T,
            levels=[0], colors=['k'], linestyles=['--']
        )

        ax.legend(proxies_for_labels, legend_for_labels, loc='upper right')

        if verbose:
            plt.show(block=False)

        return ax

    def plot_value_function_against_truth(
            self,
            reference_slice: ArraySlice2D,
            levelset: float = [0],
            target_time: float = -10,
            verbose: bool = False,
            save_path: Optional[FilePathRelative] = None
    ):

        final_values = self.get_recent_values()

        terminal_values = self.terminal_values
        solver_settings = hj_reachability.solver.SolverSettings.with_accuracy(
            accuracy=hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH,
            value_postprocessor=ReachAvoid.from_array(terminal_values, self.reach_set)
        )

        truth = hj_step(
            dynamics=self.dynamics,
            grid=self.grid,
            solver_settings=solver_settings,
            initial_values=terminal_values,
            time_start=0.,
            time_target=target_time,
            progress_bar=True
        )

        if save_path is not None:
            np.save(construct_refine_ncbf_path(generate_unique_filename(save_path, 'npy')), np.array(truth))

        x1, x2 = np.meshgrid(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim]
        )

        proxies_for_labels = [
            plt.Rectangle((0, 0), 1, 1, fc='k', ec='w', alpha=.3),
            plt.Rectangle((0, 0), 1, 1, fc='w', ec='k', alpha=1),
            plt.Rectangle((0, 0), 1, 1, fc='b', ec='w', alpha=.5),
            plt.Rectangle((0, 0), 1, 1, fc='w', ec='b', alpha=1),
            plt.Rectangle((0, 0), 1, 1, fc='r', ec='w', alpha=.5),
            plt.Rectangle((0, 0), 1, 1, fc='w', ec='r', alpha=1),
        ]

        legend_for_labels = [
            'initial',
            'initial kernel',
            'result',
            'result kernel',
            'truth',
            'truth kernel',
        ]

        fig, ax = plt.figure(figsize=(9, 7)), plt.axes(projection='3d')
        ax.set(title='value function against truth')

        ax.plot_surface(
            x1, x2, reference_slice.get_sliced_array(self.initial_values).T,
            cmap='Greys', edgecolor='none', alpha=.3
        )
        ax.contour3D(
            x1, x2, reference_slice.get_sliced_array(self.initial_values).T,
            levels=[0], colors=['k'], linewidths=1, linestyles=['--']
        )

        ax.plot_surface(
            x1, x2, reference_slice.get_sliced_array(final_values).T,
            cmap='Blues', edgecolor='none', alpha=.5
        )
        ax.contour3D(
            x1, x2, reference_slice.get_sliced_array(final_values).T,
            levels=[0], colors=['b'], linewidths=1
        )

        ax.plot_surface(
            x1, x2, reference_slice.get_sliced_array(truth).T,
            cmap='Reds', edgecolor='none', alpha=.5
        )
        ax.contour3D(
            x1, x2, reference_slice.get_sliced_array(truth).T,
            levels=levelset, colors=['r'], linewidths=1, linestyles=['--']
        )

        ax.legend(proxies_for_labels, legend_for_labels, loc='upper right')
        ax.set_xlabel(reference_slice.free_dim_1.name)
        ax.set_ylabel(reference_slice.free_dim_2.name)
        ax.set_zlabel('value')
        ax.set_zlim(bottom=-10)

        if verbose:
            plt.show(block=False)

        return fig, ax

    def plot_result_and_hjr_avoiding_result(
            self,
            reference_slice: ArraySlice2D,
            target_time: float = -10,
            verbose: bool = False,
            save_path: Optional[FilePathRelative] = None
    ):

        final_values = self.get_recent_values()

        terminal_values = self.get_recent_values()
        solver_settings = hj_reachability.solver.SolverSettings.with_accuracy(
            accuracy=hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH,
            value_postprocessor=ReachAvoid.from_array(terminal_values, self.reach_set)
        )

        truth = hj_step(
            dynamics=self.dynamics,
            grid=self.grid,
            solver_settings=solver_settings,
            initial_values=terminal_values,
            time_start=0.,
            time_target=target_time,
            progress_bar=True
        )

        if save_path is not None:
            np.save(construct_refine_ncbf_path(generate_unique_filename(save_path, 'npy')), np.array(truth))

        x1, x2 = np.meshgrid(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim]
        )

        proxies_for_labels = [
            plt.Rectangle((0, 0), 1, 1, fc='k', ec='w', alpha=.3),
            plt.Rectangle((0, 0), 1, 1, fc='w', ec='k', alpha=1),
            plt.Rectangle((0, 0), 1, 1, fc='b', ec='w', alpha=.5),
            plt.Rectangle((0, 0), 1, 1, fc='w', ec='b', alpha=1),
            plt.Rectangle((0, 0), 1, 1, fc='r', ec='w', alpha=.5),
            plt.Rectangle((0, 0), 1, 1, fc='w', ec='r', alpha=1),
        ]

        legend_for_labels = [
            'initial',
            'initial kernel',
            'result',
            'result kernel',
            'continued globally',
            'continued globally kernel',
        ]

        fig, ax = plt.figure(figsize=(9, 7)), plt.axes(projection='3d')
        ax.set(title='result, then continued with vanilla hjr')

        ax.plot_surface(
            x1, x2, reference_slice.get_sliced_array(self.initial_values).T,
            cmap='Greys', edgecolor='none', alpha=.3
        )
        ax.contour3D(
            x1, x2, reference_slice.get_sliced_array(self.initial_values).T,
            levels=[0], colors=['k'], linewidths=1, linestyles=['--']
        )

        ax.plot_surface(
            x1, x2, reference_slice.get_sliced_array(final_values).T,
            cmap='Blues', edgecolor='none', alpha=.5
        )
        ax.contour3D(
            x1, x2, reference_slice.get_sliced_array(final_values).T,
            levels=[0], colors=['b'], linewidths=1
        )

        ax.plot_surface(
            x1, x2, reference_slice.get_sliced_array(truth).T,
            cmap='Reds', edgecolor='none', alpha=.5
        )
        ax.contour3D(
            x1, x2, reference_slice.get_sliced_array(truth).T,
            levels=[0], colors=['r'], linewidths=1, linestyles=['--']
        )

        ax.legend(proxies_for_labels, legend_for_labels, loc='upper right')
        ax.set_xlabel(reference_slice.free_dim_1.name)
        ax.set_ylabel(reference_slice.free_dim_2.name)
        ax.set_zlabel('value')
        ax.set_zlim(bottom=-10)

        if verbose:
            plt.show(block=False)

        return fig, ax

    def plot_where_changed(self, reference_slice: ArraySlice2D, verbose: bool = False):
        final_values = self.get_recent_values()
        initial_values = self.initial_values

        where_changed = ~jnp.isclose(final_values, initial_values, atol=1e-3)

        x1, x2 = np.meshgrid(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim]
        )

        fig, ax = plt.figure(figsize=(9, 7)), plt.axes(projection='3d')
        ax.set(title='where changed')

        ax.contourf3D(
            x1, x2, ~reference_slice.get_sliced_array(where_changed).T,
            levels=[0, .9], colors=['b'], alpha=.5
        )
        ax.contourf3D(
            x1, x2, ~reference_slice.get_sliced_array(self.get_total_active_mask()).T,
            levels=[0, .9], colors=['r'], alpha=.5
        )

        ax.set_xlabel(reference_slice.free_dim_1.name)
        ax.set_ylabel(reference_slice.free_dim_2.name)

        if verbose:
            plt.show(block=False)

        return fig, ax

    def plot_value_function(
            self,
            reference_slice: ArraySlice2D,
            iteration: int = -1,
            verbose: bool = False,
            save_path: Optional[FilePathRelative] = None
    ):
        if save_path is not None:
            raise NotImplementedError('saving not implemented yet')

        values = self.iterations[iteration].computed_values

        x1, x2 = np.meshgrid(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim]
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
            x1, x2, reference_slice.get_sliced_array(values).T,
            cmap='Blues', edgecolor='none', alpha=.5
        )
        ax.contour3D(
            x1, x2, reference_slice.get_sliced_array(values).T,
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

    def plot_safe_cells(self, reference_slice: ArraySlice2D, verbose: bool = False):
        fig, ax = plt.subplots(figsize=(9, 7))

        ax.imshow(reference_slice.get_sliced_array(self.get_recent_values() >= 0).T, origin='lower')

        if verbose:
            plt.show(block=False)

        return fig, ax

    def plot_safe_cells_against_truth(self, reference_slice: ArraySlice2D, truth: Optional[ArrayNd] = None,
                                      verbose: bool = False):
        if truth is None:
            solver_settings = hj_reachability.solver.SolverSettings.with_accuracy(
                accuracy=hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH,
                value_postprocessor=ReachAvoid.from_array(self.terminal_values, self.reach_set)
            )
            truth = hj_step(
                dynamics=self.dynamics,
                grid=self.grid,
                solver_settings=solver_settings,
                initial_values=self.initial_values,
                time_start=0.,
                time_target=-10,
                progress_bar=True
            )
        truth_safe = reference_slice.get_sliced_array(truth >= 0).T
        result_safe = reference_slice.get_sliced_array(self.get_viability_kernel()).T
        both_safe = truth_safe & result_safe

        blue = np.array([0, 0, 255])
        red = np.array([255, 0, 0])
        green = np.array([0, 255, 0])
        white = np.array([255, 255, 255])

        image = np.zeros(truth_safe.shape + (3,), dtype=np.uint8)
        image[~truth_safe & ~result_safe] = white
        image[truth_safe] = red
        image[result_safe] = blue
        image[both_safe] = green

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(image, origin='lower')
        if verbose:
            plt.show(block=False)
