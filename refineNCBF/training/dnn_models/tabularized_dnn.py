from typing import Callable

import attr
import hj_reachability

from refineNCBF.utils.tables import tabularize_dnn, snap_state_to_grid_index
from refineNCBF.utils.types import ArrayNd, Vector


@attr.s(auto_attribs=True)
class TabularizedDnn(Callable):
    _table: ArrayNd
    _grid: hj_reachability.Grid

    @classmethod
    def from_dnn_and_grid(cls, dnn: Callable, grid: hj_reachability.Grid) -> 'TabularizedDnn':
        table = tabularize_dnn(dnn, grid)
        return cls(table, grid)

    def __call__(self, state: Vector) -> Vector:
        index = snap_state_to_grid_index(state, self._grid)
        controls = self._table[index].reshape((self._table.shape[-1], 1))
        return controls
