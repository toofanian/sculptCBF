import hj_reachability
import numpy as np
from matplotlib import pyplot as plt

from refineNCBF.utils.tables import tabularize_dnn
from refineNCBF.utils.visuals import ArraySlice2D, DimName
from scripts.barrier_refinement.pre_constrcuted_stuff.quadcopter_cbf import load_cbf_feb24

grid = hj_reachability.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj_reachability.sets.Box(
            # [-1.25, -13.7, -3.65, -7.52],
            # [11.25, 8.3, 3.26, 5.62]
            [0, -8, -np.pi, -10],
            [10, 8, np.pi, 10]
        ),
        shape=(51, 25, 51, 25)
    )

cbf, standardizer, certified_dict = load_cbf_feb24()
cbvf = -tabularize_dnn(dnn=cbf, grid=grid, standardizer=standardizer)

reference_slice = ArraySlice2D.from_reference_index(
    reference_index=(25, 12, 25, 12),
    free_dim_1=DimName(0, 'y'),
    free_dim_2=DimName(2, 'theta')
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
    x1, x2, reference_slice.get_sliced_array(cbvf).T,
    cmap='Blues', edgecolor='none', alpha=.5
)
ax.contour3D(
    x1, x2, reference_slice.get_sliced_array(cbvf).T,
    levels=[0], colors=['b']
)

ax.legend(proxies_for_labels, legend_for_labels, loc='upper right')
ax.set_xlabel(reference_slice.free_dim_1.name)
ax.set_ylabel(reference_slice.free_dim_2.name)

plt.show(block=False)
plt.pause(0)