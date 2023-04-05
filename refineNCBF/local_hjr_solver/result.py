import warnings
from typing import List, Optional, Union, Tuple

import attr
import dill
import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt, animation

import hj_reachability
from refineNCBF.hj_reachability_interface.hj_step import hj_step
from refineNCBF.hj_reachability_interface.hj_value_postprocessors import ReachAvoid
from refineNCBF.optimized_dp_interface.odp_dynamics import OdpDynamics
from refineNCBF.utils.files import (
    FilePathRelative,
    check_if_file_exists,
    construct_refine_ncbf_path,
    generate_unique_filename,
)
from refineNCBF.utils.types import MaskNd, ArrayNd
from refineNCBF.utils.visuals import ArraySlice2D, ArraySlice1D

warnings.simplefilter(action="ignore", category=FutureWarning)


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
    # blurbs: List[str] = attr.ib(default=None, validator=attr.validators.optional(attr.validators.instance_of(list)))
    blurbs: List[str] = attr.ib(factory=list)

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
        reach_set: Optional[MaskNd] = None,
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

    def add_iteration(self, iteration: LocalUpdateResultIteration, blurb: str = ""):
        self.iterations.append(iteration)
        if self.blurbs is None:
            self.blurbs = []
        self.blurbs.append(blurb)

    def save(self, file_path: FilePathRelative):
        full_path = construct_refine_ncbf_path(file_path)
        check_if_file_exists(full_path)
        with open(full_path, "wb") as f:
            dill.dump(self, f)

    def get_middle_index(self) -> Tuple[int, ...]:
        return tuple([pts // 2 for pts in self.grid.shape])

    @staticmethod
    def load(file_path: FilePathRelative) -> "LocalUpdateResult":
        try:
            full_path = construct_refine_ncbf_path(file_path)
        except FileNotFoundError:
            full_path = file_path
        with open(full_path, "rb") as f:
            cls = dill.load(f)
        return cls

    def max_diff(self):
        max_diff = jnp.max(jnp.abs(self.get_recent_values() - self.get_previous_values()))
        return max_diff

    def get_previous_values(self, iteration: int = -1):
        if len(self) > 1 and iteration > 0:
            return self.iterations[iteration - 1].computed_values
        else:
            return self.initial_values

    def get_pending_seed_set(self) -> MaskNd:
        if len(self.iterations) == 0:
            return self.seed_set
        else:
            return self.iterations[-1].active_set_post_filtered

    def get_recent_set_for_compute(self) -> MaskNd:
        if len(self) == 0:
            return self.seed_set
        else:
            return self.iterations[-1].active_set_expanded

    def get_total_active_count(self, up_to_iteration: int) -> int:
        return sum(
            [jnp.count_nonzero(iteration.active_set_expanded) for iteration in self.iterations[:up_to_iteration]]
        )

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
        ax.plot(
            self.grid.coordinate_vectors[ref_index.free_dim_1.dim],
            ref_index.get_sliced_array(self.initial_values),
        )
        for iteration in self.iterations:
            ax.plot(
                self.grid.coordinate_vectors[ref_index.free_dim_1.dim],
                ref_index.get_sliced_array(iteration.computed_values),
            )

        ax.set_xlabel(f"{ref_index.free_dim_1.name}")
        ax.set_ylabel(f"value")

        plt.show(block=False)

    def create_gif(
        self,
        reference_slice: Union[ArraySlice2D, ArraySlice1D],
        verbose: bool = True,
        save_path: Optional[FilePathRelative] = None,
    ):
        if isinstance(reference_slice, ArraySlice1D):
            reference_slice = ArraySlice2D.from_array_slice_1d(reference_slice)

        proxies_for_labels = [
            plt.Rectangle((0, 0), 1, 1, fc="r", ec="w", alpha=0.3),
            plt.Rectangle((0, 0), 1, 1, fc="g", ec="w", alpha=0.3),
            plt.Rectangle((0, 0), 1, 1, fc="k", ec="w", alpha=0.3),
            plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", alpha=0.7),
            plt.Rectangle((0, 0), 1, 1, fc="b", ec="w", alpha=0.2),
            plt.Rectangle((0, 0), 1, 1, fc="b", ec="w", alpha=0.4),
            plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", linestyle="--"),
        ]
        legend_for_labels = [
            "avoid set",
            "reach set",
            "seed set",
            "initial zero levelset",
            "active set",
            "changed set",
            "running viability kernel",
        ]

        def render_iteration(i: int):
            ax.clear()

            ax.set(
                title=f"iteration: {i}, total active: {self.get_total_active_count(i)} of {self.avoid_set.size} \nSliced at {reference_slice.slice_string}"
            )
            ax.set_xlabel(reference_slice.free_dim_1.name)
            ax.set_ylabel(reference_slice.free_dim_2.name)

            # always have reach and avoid set up
            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.avoid_set).T,
                levels=[0, 0.5],
                colors=["r"],
                alpha=0.3,
            )

            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.reach_set).T,
                levels=[0, 0.5],
                colors=["g"],
                alpha=0.3,
            )

            # always have seed set up
            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.seed_set).T,
                levels=[0, 0.5],
                colors=["k"],
                alpha=0.3,
            )

            # always have initial zero levelset up
            ax.contour(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                reference_slice.get_sliced_array(self.initial_values).T,
                levels=[0],
                colors=["k"],
                alpha=0.7,
            )

            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.iterations[i].active_set_expanded).T,
                levels=[0, 0.5],
                colors=["b"],
                alpha=0.2,
            )

            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.iterations[i].active_set_post_filtered).T,
                levels=[0, 0.5],
                colors=["b"],
                alpha=0.2,
            )

            values = self.initial_values if i == 0 else self.iterations[i - 1].computed_values
            ax.contour(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                reference_slice.get_sliced_array(values).T,
                levels=[0],
                colors=["k"],
                linestyles=["--"],
            )

            ax.legend(proxies_for_labels, legend_for_labels, loc="upper left")

            return ax

        fig, ax = plt.subplots(figsize=(9, 7))
        anim = animation.FuncAnimation(fig, render_iteration, frames=len(self), interval=100)

        if save_path is not None:
            if not check_if_file_exists(save_path):
                anim.save(construct_refine_ncbf_path(save_path), writer="imagemagick", fps=4)
            else:
                print(f"file {save_path} already exists, not saving animation")

        if verbose:
            plt.show(block=False)

    def create_gif_3d(
        self,
        reference_slice: Union[ArraySlice2D, ArraySlice1D],
        verbose: bool = True,
        save_path: Optional[FilePathRelative] = None,
    ):
        if isinstance(reference_slice, ArraySlice1D):
            reference_slice = ArraySlice2D.from_array_slice_1d(reference_slice)

        proxies_for_labels = [
            plt.Rectangle((0, 0), 1, 1, fc="b", ec="w", alpha=0.5),
            plt.Rectangle((0, 0), 1, 1, fc="w", ec="b", alpha=1),
        ]

        legend_for_labels = [
            "result",
            "result viability kernel",
        ]

        fig, ax = plt.subplots(figsize=(9, 7)), plt.axes(projection="3d")

        def render_iteration(i: int):
            values = self.iterations[i].computed_values

            x1, x2 = np.meshgrid(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            )
            ax.set(title="value function")

            ax.plot_surface(
                x1, x2, reference_slice.get_sliced_array(values).T, cmap="Blues", edgecolor="none", alpha=0.5
            )
            ax.contour3D(x1, x2, reference_slice.get_sliced_array(values).T, levels=[0], colors=["b"])

            ax.contour3D(
                x1,
                x2,
                reference_slice.get_sliced_array(self.initial_values).T,
                levels=[0],
                colors=["k"],
                linestyles=["--"],
            )

            ax.legend(proxies_for_labels, legend_for_labels, loc="upper right")
            ax.set_xlabel(reference_slice.free_dim_1.name)
            ax.set_ylabel(reference_slice.free_dim_2.name)

            return ax

        fig, ax = plt.subplots(figsize=(9, 7))
        anim = animation.FuncAnimation(fig, render_iteration, frames=len(self), interval=100)

        if save_path is not None:
            if not check_if_file_exists(save_path):
                anim.save(construct_refine_ncbf_path(save_path), writer="imagemagick", fps=4)
            else:
                print(f"file {save_path} already exists, not saving animation")

        if verbose:
            plt.show(block=False)

    def render_iteration(
        self, iteration: int, reference_slice: ArraySlice2D, verbose: bool = True, save_fig: bool = False
    ):
        fig, ax = plt.subplots(figsize=(9, 7))

        proxies_for_labels = [
            plt.Rectangle((0, 0), 1, 1, fc="r", ec="w", alpha=0.3),
            plt.Rectangle((0, 0), 1, 1, fc="g", ec="w", alpha=0.3),
            plt.Rectangle((0, 0), 1, 1, fc="k", ec="w", alpha=0.3),
            plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", alpha=0.7),
            plt.Rectangle((0, 0), 1, 1, fc="b", ec="w", alpha=0.2),
            plt.Rectangle((0, 0), 1, 1, fc="b", ec="w", alpha=0.4),
            plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", linestyle="--"),
        ]

        legend_for_labels = [
            "avoid set",
            "reach set",
            "seed set",
            "initial zero levelset",
            "active set",
            "changed set",
            "running viability kernel",
        ]

        ax.set(title=f"iteration: {iteration} of {len(self)}")

        ax.set_xlabel(reference_slice.free_dim_1.name)
        ax.set_ylabel(reference_slice.free_dim_2.name)

        # always have reach and avoid set up
        ax.contourf(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            ~reference_slice.get_sliced_array(self.avoid_set).T,
            levels=[0, 0.5],
            colors=["r"],
            alpha=0.3,
        )

        ax.contourf(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            ~reference_slice.get_sliced_array(self.reach_set).T,
            levels=[0, 0.5],
            colors=["g"],
            alpha=0.3,
        )

        # always have seed set up
        ax.contourf(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            ~reference_slice.get_sliced_array(self.seed_set).T,
            levels=[0, 0.5],
            colors=["k"],
            alpha=0.3,
        )

        # always have initial zero levelset up
        ax.contour(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            reference_slice.get_sliced_array(self.initial_values).T,
            levels=[0],
            colors=["k"],
            alpha=0.7,
        )

        ax.contourf(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            ~reference_slice.get_sliced_array(self.iterations[iteration].active_set_expanded).T,
            levels=[0, 0.5],
            colors=["b"],
            alpha=0.2,
        )

        ax.contourf(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            ~reference_slice.get_sliced_array(self.iterations[iteration].active_set_post_filtered).T,
            levels=[0, 0.5],
            colors=["b"],
            alpha=0.2,
        )

        values = self.initial_values if iteration == 0 else self.iterations[iteration - 1].computed_values
        ax.contour(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            reference_slice.get_sliced_array(values).T,
            levels=[0],
            colors=["k"],
            linestyles=["--"],
        )

        ax.legend(proxies_for_labels, legend_for_labels, loc="upper right")

        if save_fig:
            plt.savefig(
                construct_refine_ncbf_path(
                    generate_unique_filename(f"data/visuals/render_iteration_{iteration}", "png")
                )
            )

        if verbose:
            plt.show(block=False)

        return ax

    def plot_value_function_against_truth(
        self,
        reference_slice: ArraySlice2D,
        levelset: float = [0],
        target_time: float = -10,
        verbose: bool = False,
        save_path: Optional[FilePathRelative] = None,
    ):

        final_values = self.get_recent_values()

        terminal_values = self.terminal_values
        solver_settings = hj_reachability.solver.SolverSettings.with_accuracy(
            accuracy=hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH,
            value_postprocessor=ReachAvoid.from_array(terminal_values, self.reach_set),
        )

        truth = hj_step(
            dynamics=self.dynamics,
            grid=self.grid,
            solver_settings=solver_settings,
            initial_values=terminal_values,
            time_start=0.0,
            time_target=target_time,
            progress_bar=True,
        )

        if save_path is not None:
            np.save(construct_refine_ncbf_path(generate_unique_filename(save_path, "npy")), np.array(truth))

        x1, x2 = np.meshgrid(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
        )

        proxies_for_labels = [
            plt.Rectangle((0, 0), 1, 1, fc="k", ec="w", alpha=0.3),
            plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", alpha=1),
            plt.Rectangle((0, 0), 1, 1, fc="b", ec="w", alpha=0.5),
            plt.Rectangle((0, 0), 1, 1, fc="w", ec="b", alpha=1),
            plt.Rectangle((0, 0), 1, 1, fc="r", ec="w", alpha=0.5),
            plt.Rectangle((0, 0), 1, 1, fc="w", ec="r", alpha=1),
        ]

        legend_for_labels = [
            "initial",
            "initial kernel",
            "result",
            "result kernel",
            "truth",
            "truth kernel",
        ]

        fig, ax = plt.subplots(figsize=(9, 7)), plt.axes(projection="3d")
        ax.set(title="value function against truth")

        ax.plot_surface(
            x1, x2, reference_slice.get_sliced_array(self.initial_values).T, cmap="Greys", edgecolor="none", alpha=0.3
        )
        ax.contour3D(
            x1,
            x2,
            reference_slice.get_sliced_array(self.initial_values).T,
            levels=[0],
            colors=["k"],
            linewidths=1,
            linestyles=["--"],
        )

        ax.plot_surface(
            x1, x2, reference_slice.get_sliced_array(final_values).T, cmap="Blues", edgecolor="none", alpha=0.5
        )
        ax.contour3D(x1, x2, reference_slice.get_sliced_array(final_values).T, levels=[0], colors=["b"], linewidths=1)

        ax.plot_surface(x1, x2, reference_slice.get_sliced_array(truth).T, cmap="Reds", edgecolor="none", alpha=0.5)
        ax.contour3D(
            x1,
            x2,
            reference_slice.get_sliced_array(truth).T,
            levels=levelset,
            colors=["r"],
            linewidths=1,
            linestyles=["--"],
        )

        ax.legend(proxies_for_labels, legend_for_labels, loc="upper right")
        ax.set_xlabel(reference_slice.free_dim_1.name)
        ax.set_ylabel(reference_slice.free_dim_2.name)
        ax.set_zlabel("value")
        ax.set_zlim(bottom=-10)

        if verbose:
            plt.show(block=False)

        return fig, ax

    def plot_result_and_hjr_avoiding_result(
        self,
        reference_slice: ArraySlice2D,
        target_time: float = -10,
        verbose: bool = False,
        save_path: Optional[FilePathRelative] = None,
    ):

        final_values = self.get_recent_values()

        terminal_values = self.get_recent_values()
        solver_settings = hj_reachability.solver.SolverSettings.with_accuracy(
            accuracy=hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH,
            value_postprocessor=ReachAvoid.from_array(terminal_values, self.reach_set),
        )

        truth = hj_step(
            dynamics=self.dynamics,
            grid=self.grid,
            solver_settings=solver_settings,
            initial_values=terminal_values,
            time_start=0.0,
            time_target=target_time,
            progress_bar=True,
        )

        if save_path is not None:
            np.save(construct_refine_ncbf_path(generate_unique_filename(save_path, "npy")), np.array(truth))

        x1, x2 = np.meshgrid(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
        )

        proxies_for_labels = [
            plt.Rectangle((0, 0), 1, 1, fc="k", ec="w", alpha=0.3),
            plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", alpha=1),
            plt.Rectangle((0, 0), 1, 1, fc="b", ec="w", alpha=0.5),
            plt.Rectangle((0, 0), 1, 1, fc="w", ec="b", alpha=1),
            plt.Rectangle((0, 0), 1, 1, fc="r", ec="w", alpha=0.5),
            plt.Rectangle((0, 0), 1, 1, fc="w", ec="r", alpha=1),
        ]

        legend_for_labels = [
            "initial",
            "initial kernel",
            "result",
            "result kernel",
            "continued globally",
            "continued globally kernel",
        ]

        fig, ax = plt.subplots(figsize=(9, 7)), plt.axes(projection="3d")
        ax.set(title="result, then continued with vanilla hjr")

        ax.plot_surface(
            x1, x2, reference_slice.get_sliced_array(self.initial_values).T, cmap="Greys", edgecolor="none", alpha=0.3
        )
        ax.contour3D(
            x1,
            x2,
            reference_slice.get_sliced_array(self.initial_values).T,
            levels=[0],
            colors=["k"],
            linewidths=1,
            linestyles=["--"],
        )

        ax.plot_surface(
            x1, x2, reference_slice.get_sliced_array(final_values).T, cmap="Blues", edgecolor="none", alpha=0.5
        )
        ax.contour3D(x1, x2, reference_slice.get_sliced_array(final_values).T, levels=[0], colors=["b"], linewidths=1)

        ax.plot_surface(x1, x2, reference_slice.get_sliced_array(truth).T, cmap="Reds", edgecolor="none", alpha=0.5)
        ax.contour3D(
            x1, x2, reference_slice.get_sliced_array(truth).T, levels=[0], colors=["r"], linewidths=1, linestyles=["--"]
        )

        ax.legend(proxies_for_labels, legend_for_labels, loc="upper right")
        ax.set_xlabel(reference_slice.free_dim_1.name)
        ax.set_ylabel(reference_slice.free_dim_2.name)
        ax.set_zlabel("value")
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
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
        )

        fig, ax = plt.subplots(figsize=(9, 7)), plt.axes(projection="3d")
        ax.set(title="where changed")

        ax.contourf3D(
            x1, x2, ~reference_slice.get_sliced_array(where_changed).T, levels=[0, 0.9], colors=["b"], alpha=0.5
        )
        ax.contourf3D(
            x1,
            x2,
            ~reference_slice.get_sliced_array(self.get_total_active_mask()).T,
            levels=[0, 0.9],
            colors=["r"],
            alpha=0.5,
        )

        ax.set_xlabel(reference_slice.free_dim_1.name)
        ax.set_ylabel(reference_slice.free_dim_2.name)

        if verbose:
            plt.show(block=False)

        return fig, ax

    def plot_value_function(
        self,
        reference_slice: ArraySlice2D,
        iteration: Optional[Union[int, List[int]]] = -1,
        verbose: bool = False,
        save_path: Optional[FilePathRelative] = None,
    ):
        if save_path is not None:
            raise NotImplementedError("saving not implemented yet")

        if isinstance(iteration, list):
            values = []
            for itr in iteration:
                values.append(self.iterations[itr].computed_values)
        else:
            values = self.iterations[iteration].computed_values

        x1, x2 = np.meshgrid(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
        )

        proxies_for_labels = [
            plt.Rectangle((0, 0), 1, 1, fc="b", ec="w", alpha=0.5),
            plt.Rectangle((0, 0), 1, 1, fc="w", ec="b", alpha=1),
        ]

        legend_for_labels = [
            "result",
            "result viability kernel",
        ]

        fig, ax = plt.subplots(figsize=(9, 7)), plt.axes(projection="3d")
        ax.set(title="value function")

        ax.plot_surface(x1, x2, reference_slice.get_sliced_array(values).T, cmap="Blues", edgecolor="none", alpha=0.5)
        ax.contour3D(x1, x2, reference_slice.get_sliced_array(values).T, levels=[0], colors=["b"])

        ax.contour3D(
            x1, x2, reference_slice.get_sliced_array(self.initial_values).T, levels=[0], colors=["k"], linestyles=["--"]
        )

        ax.legend(proxies_for_labels, legend_for_labels, loc="upper right")
        ax.set_xlabel(reference_slice.free_dim_1.name)
        ax.set_ylabel(reference_slice.free_dim_2.name)

        if verbose:
            plt.show(block=False)

        return fig, ax

    def plot_safe_cells(self, reference_slice: ArraySlice2D, verbose: bool = False):
        fig, ax = plt.subplots(figsize=(9, 7))

        ax.imshow(reference_slice.get_sliced_array(self.get_recent_values() >= 0).T, origin="lower")

        if verbose:
            plt.show(block=False)

        return fig, ax

    def plot_safe_cells_against_truth(
        self, reference_slice: ArraySlice2D, truth: Optional[ArrayNd] = None, verbose: bool = False
    ):
        if truth is None:
            solver_settings = hj_reachability.solver.SolverSettings.with_accuracy(
                accuracy=hj_reachability.solver.SolverAccuracyEnum.VERY_HIGH,
                value_postprocessor=ReachAvoid.from_array(self.terminal_values, self.reach_set),
            )
            truth = hj_step(
                dynamics=self.dynamics,
                grid=self.grid,
                solver_settings=solver_settings,
                initial_values=self.initial_values,
                time_start=0.0,
                time_target=-10,
                progress_bar=True,
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
        ax.imshow(image, origin="lower")
        if verbose:
            plt.show(block=False)
        else:
            return fig, ax

    def plot_algorithm(
        self,
        iteration: int,
        reference_slice: ArraySlice2D,
        vis_type: str = "before",  # "before", "after", "change"
        ax: Optional[plt.Axes] = None,
        legend: bool = False,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ):
        plt.rcParams["text.usetex"] = True

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.figure

        if vis_type == "before":
            # show active set Q_k and boundary of V_k before update
            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.avoid_set).T,
                levels=[0, 0.5],
                colors=["r"],
                alpha=0.3,
            )

            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.iterations[iteration].active_set_expanded).T,
                levels=[0, 0.5],
                colors=["b"],
                alpha=0.3,
            )

            ax.contour(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                reference_slice.get_sliced_array(self.get_previous_values(iteration)).T,
                levels=[0],
                colors=["k"],
                alpha=0.7,
                linewidths=10,
            )

            proxies_for_labels = [
                plt.Rectangle((0, 0), 1, 1, fc="r", ec="w", alpha=0.3),
                plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", linestyle="-"),
                plt.Rectangle((0, 0), 1, 1, fc="b", ec="w", alpha=0.3),
            ]
            legend_for_labels = [r"$L$, failure set", r"$V_{k} \approx 0$, unsafe boundary", r"$Q_k$, active set"]

        elif vis_type == "after":
            # show active set Q_k and boundary V_k+1 after update

            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.avoid_set).T,
                levels=[0, 0.5],
                colors=["r"],
                alpha=0.3,
            )

            slice_of_interest = reference_slice.get_sliced_array(self.iterations[iteration].active_set_expanded).T
            second_slice = reference_slice.get_sliced_array(self.iterations[iteration].active_set_post_filtered).T
            slice_to_vis = np.logical_or(second_slice, slice_of_interest)
            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~slice_to_vis,
                levels=[0, 0.5],
                colors=["b"],
                alpha=0.3,
            )

            ax.contour(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                reference_slice.get_sliced_array(self.get_previous_values(iteration)).T,
                levels=[0],
                colors=["k"],
                alpha=0.7,
                linestyles=["--"],
                linewidths=10,
            )

            ax.contour(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                reference_slice.get_sliced_array(self.iterations[iteration].computed_values).T,
                levels=[0],
                colors=["k"],
                alpha=0.7,
                linewidths=10,
            )

            proxies_for_labels = [
                plt.Rectangle((0, 0), 1, 1, fc="r", ec="w", alpha=0.3),
                plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", linestyle="-"),
                plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", linestyle="--"),
                plt.Rectangle((0, 0), 1, 1, fc="b", ec="w", alpha=0.4),
                plt.Rectangle((0, 0), 1, 1, fc="b", ec="w", alpha=0.2),
            ]
            legend_for_labels = [
                r"$L$, failure set",
                r"$V^{(k+1)} \approx 0$, updated unsafe boundary",
                r"$V^{(k)} \approx 0$, unsafe boundary",
                r"$Q^{k^-}$, leaky active set",
                r"$Q^k$, active set",
            ]

        elif vis_type == "change":
            # show active set Q_k and neighbor cells on boundary and boundary V_k+1 after update

            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(self.avoid_set).T,
                levels=[0, 0.5],
                colors=["r"],
                alpha=0.3,
            )
            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(
                    self.iterations[iteration].active_set_expanded
                    & ~(
                        ~self.iterations[iteration].active_set_expanded
                        & self.iterations[iteration + 1].active_set_expanded
                    )
                    & ~(
                        self.iterations[iteration].active_set_expanded
                        & ~self.iterations[iteration + 1].active_set_expanded
                    )
                ).T,
                levels=[0, 0.5],
                colors=["b"],
                alpha=0.3,
            )

            # removed cells
            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(
                    (
                        self.iterations[iteration].active_set_expanded
                        & ~self.iterations[iteration + 1].active_set_expanded
                    )
                ).T,
                levels=[0, 0.5],
                colors=["b"],
                alpha=0.15,
            )

            # added cells
            ax.contourf(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                ~reference_slice.get_sliced_array(
                    (
                        ~self.iterations[iteration].active_set_expanded
                        & self.iterations[iteration + 1].active_set_expanded
                    )
                ).T,
                levels=[0, 0.5],
                colors=["b"],
                alpha=0.45,
            )

            ax.contour(
                self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
                self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
                reference_slice.get_sliced_array(self.iterations[iteration].computed_values).T,
                levels=[0],
                colors=["k"],
                alpha=0.7,
                linewidths=10.0,
            )

            proxies_for_labels = [
                plt.Rectangle((0, 0), 1, 1, fc="r", ec="w", alpha=0.3),
                plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", linestyle="-"),
                plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", linestyle="--"),
                plt.Rectangle((0, 0), 1, 1, fc="b", ec="w", alpha=0.3),
                plt.Rectangle((0, 0), 1, 1, fc="g", ec="w", alpha=0.3),
                plt.Rectangle((0, 0), 1, 1, fc="y", ec="w", alpha=0.3),
            ]
            legend_for_labels = [
                r"$L$, failure set",
                r"$V^{(k+1)} \approx 0$, updated unsafe boundary",
                r"$V^{(k)} \approx 0$, unsafe boundary",
                r"cells carried over from active set $Q_{k+1}$",
                r"cells added to active set $Q_{k+1}$",
                r"cells removed from active set $Q_{k+1}$",
            ]
        else:
            raise ValueError("Unknown vis_type {}".format(vis_type))

        # For all vis types
        # Set the position of the legend, and optionally draw the legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        # Put a legend below current axis
        if legend:
            ax.legend(
                proxies_for_labels,
                legend_for_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.05),
                fancybox=True,
                shadow=True,
                ncol=3,
            )

        # Hide X and Y axes label marks
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        # ax.set_xlim([-np.pi, -1.0])
        # ax.set_ylim([3, 9])

        return fig, ax

    def plot_kernel_accuracy_vs_hammys(
        self,
        label: str,
        title: str,
        x_scale: Optional[str] = "log",
        y_scale: Optional[str] = "log",
        compare_results: Optional[List["LocalUpdateResult"]] = None,
        compare_labels: Optional[List[str]] = None,
    ):
        """
        assumes final result is the converged kernel
        """
        plt.rcParams["text.usetex"] = True
        fig, ax = plt.subplots(figsize=(16, 9))

        final_kernel = self.get_viability_kernel()
        initial_kernel = self.initial_values >= 0
        viable_count = np.count_nonzero(initial_kernel & ~final_kernel)

        inaccurate_unsafe_counts = [viable_count]
        hammies = [1]
        for iteration in self.iterations:
            inaccurate_unsafe_counts.append(np.count_nonzero((iteration.computed_values >= 0) & ~final_kernel))
            hammies.append(np.count_nonzero(iteration.active_set_expanded))

        inaccurate_unsafe_fractions = [
            inaccurate_unsafe_count / viable_count for inaccurate_unsafe_count in inaccurate_unsafe_counts
        ]

        running_hammies = [sum(hammies[: i + 1]) for i in range(len(hammies))]

        ax.plot(running_hammies, inaccurate_unsafe_fractions, label=label, marker="x")

        if compare_results is not None:
            if type(compare_results) is not list:
                compare_results = [compare_results]
            if type(compare_labels) is not list:
                compare_labels = [compare_labels]
            for i, compare_result in enumerate(compare_results):
                final_kernel = compare_result.get_viability_kernel()
                initial_kernel = compare_result.initial_values >= 0
                viable_count = np.count_nonzero(initial_kernel & ~final_kernel)

                inaccurate_unsafe_counts = [viable_count]
                hammies = [1]
                for iteration in compare_result.iterations:
                    inaccurate_unsafe_counts.append(np.count_nonzero((iteration.computed_values >= 0) & ~final_kernel))
                    hammies.append(np.count_nonzero(iteration.active_set_expanded))

                inaccurate_unsafe_fractions = [
                    inaccurate_unsafe_count / viable_count for inaccurate_unsafe_count in inaccurate_unsafe_counts
                ]

                running_hammies = [sum(hammies[: i + 1]) for i in range(len(hammies))]

                ax.plot(running_hammies, inaccurate_unsafe_fractions, label=compare_labels[i], marker="x")

        ax.set_title(title)
        ax.set_xscale(x_scale)
        ax.set_xlim(left=100) if y_scale == "log" else ax.set_xlim(left=1)
        ax.set_yscale(y_scale)
        ax.set_ylim(bottom=0.001, top=1)
        ax.set_xlabel("Total Hamiltonians Computed")
        ax.set_ylabel("Fraction of Unsafe States in Running Kernel")

        ax.legend()

        return fig, ax

    @staticmethod
    def plot_wrongly_safe_cells(
        results: List["LocalUpdateResult"],
        labels: List[str],
        ground_truth: ArrayNd,
        title: Optional[str] = "",
        x_scale: Optional[str] = "linear",
        y_scale: Optional[str] = "log",
    ):
        results = [results] if type(results) is not list else results
        labels = [labels] if type(labels) is not list else labels
        fig, ax = plt.subplots(figsize=(16, 9))
        true_safe = ground_truth >= 0
        for i, results in enumerate(results):
            wrongly_safe_counts = []
            initially_safe = results.initial_values >= 0
            wrongly_safe_counts.append(np.count_nonzero(initially_safe & ~true_safe))
            for iteration in results.iterations:
                wrongly_safe_counts.append(np.count_nonzero((iteration.computed_values >= 0) & ~true_safe))
            wrongly_safe_counts = np.array(wrongly_safe_counts)
            wrongly_safe_counts = wrongly_safe_counts / float(true_safe.size)
            ax.plot(wrongly_safe_counts, label=labels[i])
        ax.set_title("Wrongly Safe Cells")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fraction of Cells (of global grid) that are wrongly safe")
        ax.legend()
        ax.set_title(title)
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

        return fig, ax

    @staticmethod
    def plot_nbr_safe_cells(
        results: List["LocalUpdateResult"],
        labels: List[str],
        title: Optional[str] = "Safe cells",
        x_scale: Optional[str] = "linear",
        y_scale: Optional[str] = "log",
        x_type: Optional[str] = "iteration",
    ):
        results = [results] if type(results) is not list else results
        labels = [labels] if type(labels) is not list else labels
        fig, ax = plt.subplots(figsize=(16, 9))
        for i, results in enumerate(results):
            safe_counts = []
            hammies = [1]
            initially_safe = results.initial_values >= 0
            safe_counts.append(np.count_nonzero(initially_safe))
            for iteration in results.iterations:
                safe_counts.append(np.count_nonzero((iteration.computed_values >= 0)))
                hammies.append(np.count_nonzero(iteration.active_set_expanded))
            safe_counts = np.array(safe_counts)
            running_hammies = [sum(hammies[: i + 1]) for i in range(len(hammies))]
            safe_counts = safe_counts / float(iteration.computed_values.size)
            if x_type == "hammy":
                ax.plot(running_hammies, safe_counts, label=labels[i], marker="x")
                ax.set_xlabel("Total Hamiltonians Computed")
            else:
                ax.plot(safe_counts, label=labels[i], marker="x")
                ax.set_xlabel("Iteration")
        ax.set_title(title)

        ax.set_ylabel("Fraction of Cells (of global grid) that are safe")
        ax.legend()
        ax.set_title(title)
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

        return fig, ax

    def plot_value_function_comparison(
        self,
        reference_slice: ArraySlice2D,
        title: str,
        label: str,
        iteration: int = -1,
        comparison_result: Optional["LocalUpdateResult"] = None,
        comparison_iteration: int = -1,
        comparison_label: Optional[str] = None,
        legend: bool = True,
        verbose: bool = False,
        save_fig: bool = False,
        ax: Optional[plt.Axes] = None,
    ):
        plt.rcParams["text.usetex"] = True
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 7)), plt.axes(projection="3d")
        else:
            fig = ax.figure

        values = self.iterations[iteration].computed_values

        x1, x2 = np.meshgrid(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
        )

        ax.contour3D(
            x1,
            x2,
            reference_slice.get_sliced_array(self.initial_values).T,
            levels=[0],
            colors=["k"],
            alpha=0.5,
            linestyles=["--"],
            linewidths=5,
        )
        ax.contour(
            x1,
            x2,
            reference_slice.get_sliced_array(self.initial_values).T,
            levels=[0],
            colors=["k"],
            linestyles=["--"],
            linewidths=5,
            zdir="z",
            offset=-40,
        )

        ax.contour3D(x1, x2, reference_slice.get_sliced_array(values).T, levels=[0], colors=["b"], linewidths=5)
        ax.contour(
            x1,
            x2,
            reference_slice.get_sliced_array(values).T,
            levels=[0],
            colors=["b"],
            linewidths=5,
            zdir="z",
            offset=-40,
        )
        proxies_for_labels = [
            plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", alpha=1, linestyle="--"),
            plt.Rectangle((0, 0), 1, 1, fc="r", ec="w", alpha=0.5),
            plt.Rectangle((0, 0), 1, 1, fc="b", ec="w", alpha=0.5),
            plt.Rectangle((0, 0), 1, 1, fc="w", ec="b", alpha=1),
        ]
        legend_for_labels = [
            r"$V_0(x) \approx 0$",
            r"$L$, failure set",
            r"$V_\infty(x)$, " + label,
            r"$V_\infty(x)\approx 0$, " + label,
        ]

        if comparison_result is not None:
            comparison_values = comparison_result.iterations[comparison_iteration].computed_values

            ax.plot_surface(
                x1,
                x2,
                reference_slice.get_sliced_array(comparison_values).T,
                cmap="Greens",
                edgecolor="none",
                alpha=0.8,
                linewidth=3,
                shade=False,
            )
            ax.contour3D(
                x1,
                x2,
                reference_slice.get_sliced_array(comparison_values).T,
                levels=[0],
                colors=["g"],
                linestyles=["--"],
                linewidths=5,
            )
            ax.contour(
                x1,
                x2,
                reference_slice.get_sliced_array(comparison_values).T,
                levels=[0],
                colors=["g"],
                linestyles=["--"],
                linewidths=5,
                zdir="z",
                offset=-40,
            )

            proxies_for_labels.extend(
                [
                    plt.Rectangle((0, 0), 1, 1, fc="g", ec="w", alpha=0.5),
                    plt.Rectangle((0, 0), 1, 1, fc="w", ec="g", alpha=1),
                ]
            )
            legend_for_labels.extend(
                [
                    r"$V_\infty(x)$, " + comparison_label,
                    r"$V_\infty(x)\approx 0$, " + comparison_label,
                ]
            )
        ax.plot_surface(
            x1,
            x2,
            reference_slice.get_sliced_array(values).T,
            cmap="Blues",
            edgecolor="none",
            alpha=0.8,
            linewidth=3,
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        # Put a legend below current axis
        if legend:
            ax.legend(
                proxies_for_labels,
                legend_for_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.05),
                fancybox=True,
                shadow=True,
                ncol=3,
            )
        ax.set_xlabel(reference_slice.free_dim_1.name, labelpad=10)
        ax.set_ylabel(reference_slice.free_dim_2.name, labelpad=10)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel("Value", rotation=90, labelpad=10)

        # Change the elevation
        ax.view_init(25, -60)
        if save_fig:
            plt.savefig(
                construct_refine_ncbf_path(
                    generate_unique_filename(f'data/visuals/{title.replace(",","").replace(" ", "_")}', "pdf")
                ),
                bbox_inches="tight",
            )

        if verbose:
            plt.show(block=False)

        return fig, ax

    def plot_value_2d_comparison(
        self,
        reference_slice: ArraySlice2D,
        title: str,
        label: str,
        iteration: int = -1,
        comparison_result: Optional["LocalUpdateResult"] = None,
        comparison_iteration: int = -1,
        comparison_label: Optional[str] = None,
        legend: bool = True,
        verbose: bool = False,
        save_fig: bool = False,
        ax: Optional[plt.Axes] = None,
    ):
        plt.rcParams["text.usetex"] = True
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 3.88))
        else:
            fig = ax.figure
        add_comparison = True
        ax.contourf(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            ~reference_slice.get_sliced_array(self.avoid_set).T,
            levels=[0, 0.5],
            colors=["r"],
            alpha=0.3,
        )

        ax.contour(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            reference_slice.get_sliced_array(self.iterations[iteration].computed_values).T,
            levels=[0],
            colors=["b"],
            alpha=1,
            linewidths=5,
        )
        if comparison_result is None:
            if comparison_iteration == iteration:
                add_comparison = False
            else:
                comparison_result = self

        ax.contour(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            reference_slice.get_sliced_array(comparison_result.iterations[comparison_iteration].computed_values).T,
            levels=[0],
            colors=["g"],
            alpha=1,
            linestyles=["--"],
            linewidths=5,
        )
        ax.contour(
            self.grid.coordinate_vectors[reference_slice.free_dim_1.dim],
            self.grid.coordinate_vectors[reference_slice.free_dim_2.dim],
            reference_slice.get_sliced_array(self.initial_values).T,
            levels=[0],
            colors=["k"],
            alpha=1,
            linestyles=["--"],
            linewidths=5,
        )

        proxies_for_labels = [
            plt.Rectangle((0, 0), 1, 1, fc="r", ec="w", alpha=0.3),
            plt.Rectangle((0, 0), 1, 1, fc="w", ec="b", linestyle="-"),
            plt.Rectangle((0, 0), 1, 1, fc="w", ec="g", linestyle="-"),
        ]

        legend_for_labels = [
            r"$L$, failure set",
            r"$V_{0} \approx 0$, " + label,
            r"$V_{0} \approx 0$, " + comparison_label,
        ]

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        # Put a legend below current axis
        if legend:
            ax.legend(
                proxies_for_labels,
                legend_for_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.05),
                fancybox=True,
                shadow=True,
                ncol=3,
            )
        ax.set_xlabel(reference_slice.free_dim_1.name)
        ax.set_ylabel(reference_slice.free_dim_2.name)

        if save_fig:
            plt.savefig(
                construct_refine_ncbf_path(
                    generate_unique_filename(f'data/visuals/{title.replace(",","").replace(" ", "_")}', "pdf")
                ),
                bbox_inches="tight",
            )

        if verbose:
            plt.show(block=False)

        return fig, ax
