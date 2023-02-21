import time

import hj_reachability
import numpy as np
from matplotlib import pyplot as plt
from odp.Grid import Grid
from odp.Plots import PlotOptions
from odp.Shapes import Lower_Half_Space, Upper_Half_Space, Intersection
from odp.solver import hj_solve, HjSolver

import heterocl as hcl

from refineNCBF.utils.visuals import ArraySlice2D, DimName


class ActiveCruiseControl:
    def __init__(self, friction_coeffs, target_velocity, mass, uMode="min", dMode="max"):
        self.friction_coefficients = friction_coeffs
        self.target_velocity = target_velocity
        self.mass = mass
        self.uMode = uMode
        self.control_upper_bounds = [5000.0]
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        opt_a = hcl.scalar(self.control_upper_bounds[0], "opt_a")
        in2 = hcl.scalar(0, "in2")
        in3 = hcl.scalar(0, "in3")

        with hcl.if_(spat_deriv[1] < 0):
            opt_a[0] = -opt_a

        return opt_a[0], in2[0], in3[0]

    def opt_dstb(self, t, state, spat_deriv):
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        return d1[0], d2[0], d3[0]

    def dynamics(self, t, state, u_opt, d_opt):
        x1_dot = hcl.scalar(0, "x1_dot")
        x2_dot = hcl.scalar(0, "x2_dot")
        x3_dot = hcl.scalar(0, "x3_dot")

        x1_dot[0] = state[1]
        x2_dot[0] = -1 / self.mass * \
            (
                    self.friction_coefficients[0] +
                    self.friction_coefficients[1] *
                    state[1] +
                    self.friction_coefficients[2] *
                    state[1]*state[1]
            ) \
            + 1 / self.mass * u_opt[0]
        x3_dot[0] = self.target_velocity - state[1]
        return x1_dot[0], x2_dot[0], x3_dot[0]


g = Grid(np.array([0.0, -20.0, 20.0]), np.array([1e3, 20.0, 80.0]), 3, np.array([50, 50, 50]))
shape1 = Lower_Half_Space(g, dim=2, value=60)
shape2 = Upper_Half_Space(g, dim=2, value=40)
initial_value_f = -Intersection(shape1, shape2)
tau = [0, 5]
my_car = ActiveCruiseControl([0.1, 5.0, 0.25], 0.0, 1650, uMode="max", dMode="min")
po2 = PlotOptions(do_plot=False, plot_type="2d_plot", plotDims=[1, 2], slicesCut=[0])
compMethods = {"TargetSetMode": "minVWithV0"}

my_car_1 = my_car
g_1 = g
initial_value_f_1 = initial_value_f.copy()
tau_1 = tau
compMethods_1 = compMethods
po2_1 = po2

# active_set = initial_value_f > 0
active_set = np.ones_like(initial_value_f)

solver_1 = HjSolver()

values_1 = solver_1(my_car_1, g_1, initial_value_f_1, tau_1, compMethods_1, po2_1, saveAllTimeSteps=False, active_set=active_set)
print(
    'hello'
)
time.sleep(5)
values_2 = solver_1(my_car_1, g_1, initial_value_f_1, tau_1, compMethods_1, po2_1, saveAllTimeSteps=False, active_set=active_set)
print(
    'hello'
)
time.sleep(5)



#
# grid_hj = hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
#         domain=hj_reachability.sets.Box(
#             [0, -20, 20],
#             [1e3, 20, 80]
#         ),
#         shape=(50, 50, 50)
#     )
#
# reference_slice = ArraySlice2D.from_reference_index(
#     reference_index=(10, 0, 0),
#     free_dim_1=DimName(1, 'relative velocity'),
#     free_dim_2=DimName(2, 'relative position'),
# )
#
# x1, x2 = np.meshgrid(
#     grid_hj.coordinate_vectors[reference_slice.free_dim_1.dim],
#     grid_hj.coordinate_vectors[reference_slice.free_dim_2.dim]
# )
#
# proxies_for_labels = [
#     plt.Rectangle((0, 0), 1, 1, fc='b', ec='w', alpha=.5),
#     plt.Rectangle((0, 0), 1, 1, fc='w', ec='b', alpha=1),
# ]
#
# legend_for_labels = [
#     'result',
#     'result viability kernel',
# ]
#
# fig, ax = plt.figure(figsize=(9, 7)), plt.axes(projection='3d')
# ax.set(title='value function')
#
# ax.plot_surface(
#     x1, x2, reference_slice.get_sliced_array(values).T,
#     cmap='Blues', edgecolor='none', alpha=.5
# )
# ax.contour3D(
#     x1, x2, reference_slice.get_sliced_array(values).T,
#     levels=[0], colors=['b']
# )
#
# ax.contour3D(
#     x1, x2, reference_slice.get_sliced_array(initial_value_f).T,
#     levels=[0], colors=['k'], linestyles=['--']
# )
#
# ax.legend(proxies_for_labels, legend_for_labels, loc='upper right')
# ax.set_xlabel(reference_slice.free_dim_1.name)
# ax.set_ylabel(reference_slice.free_dim_2.name)
#
# plt.show(block=False)
# plt.pause(0)