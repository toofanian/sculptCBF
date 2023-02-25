from abc import ABC, abstractmethod
from typing import Callable

import attr
from jax import numpy as jnp

from refineNCBF.local_hjr_solver.result import LocalUpdateResult
from refineNCBF.utils.types import MaskNd, ArrayNd


@attr.s(auto_attribs=True)
class ActiveSetPostFilter(ABC, Callable):
    @abstractmethod
    def __call__(
            self,
            data: LocalUpdateResult,
            active_set_pre_filtered: MaskNd,
            active_set_expanded: MaskNd,
            values_next: ArrayNd
    ) -> MaskNd:
        ...


@attr.s(auto_attribs=True)
class RemoveWhereUnchanged(ActiveSetPostFilter):
    _atol: float
    _rtol: float

    @classmethod
    def from_parts(cls, atol: float, rtol: float):
        return cls(atol=atol, rtol=rtol)

    def __call__(
            self,
            data: LocalUpdateResult,
            active_set_pre_filtered: MaskNd,
            active_set_expanded: MaskNd,
            values_next: ArrayNd
    ) -> MaskNd:
        changed = active_set_expanded & ~jnp.isclose(
            data.get_recent_values(), values_next,
            atol=self._atol, rtol=self._rtol
        )
        return changed


@attr.s(auto_attribs=True)
class RemoveWhereNonNegativeHamiltonian(ActiveSetPostFilter):
    hamitonian_atol: float = 1e-3

    @classmethod
    def from_parts(cls, hamiltonian_atol: float = 1e-3):
        return cls(hamiltonian_atol)

    def __call__(
            self,
            data: LocalUpdateResult,
            active_set_pre_filtered: MaskNd,
            active_set_expanded: MaskNd,
            values_next: ArrayNd
    ) -> MaskNd:
        hamiltonian = values_next - data.get_recent_values()
        negative_hamiltonian = hamiltonian < -self.hamitonian_atol
        return negative_hamiltonian & active_set_expanded


@attr.s(auto_attribs=True)
class NoPostFilter(ActiveSetPostFilter):
    @classmethod
    def from_parts(cls):
        return cls()

    def __call__(
            self,
            data: LocalUpdateResult,
            active_set_pre_filtered: MaskNd,
            active_set_expanded: MaskNd,
            values_next: ArrayNd
    ) -> MaskNd:
        return active_set_expanded


@attr.s(auto_attribs=True)
class RemoveWhereOscillating(ActiveSetPostFilter):
    """
    WARNING: possibly incomplete implementation
    # TODO remove this object?
    """
    _history_length: int
    _std_threshold: float

    @classmethod
    def from_parts(cls, history_length: int, std_threshold: float):
        return cls(history_length, std_threshold)

    def __call__(
            self,
            data: LocalUpdateResult,
            active_set_pre_filtered: MaskNd,
            active_set_expanded: MaskNd,
            values_next: ArrayNd
    ) -> MaskNd:
        recent_values_list = data.get_recent_values_list(span=self._history_length)
        recent_values_array = jnp.stack(recent_values_list, axis=-1)
        values_std = jnp.std(recent_values_array, axis=-1)
        values_mean = jnp.mean(recent_values_array, axis=-1)

        where_narrow_std = values_std < self._std_threshold
        where_close_to_mean = jnp.abs(values_next - values_mean) <= values_std
        oscillating = where_narrow_std & where_close_to_mean

        return active_set_expanded & ~oscillating


@attr.s(auto_attribs=True)
class RemoveWhereUnchangedOrOscillating(ActiveSetPostFilter):
    """
    WARNING: possibly incomplete implementation
    # TODO remove this object?
    """
    _atol: float
    _rtol: float
    _history_length: int
    _std_threshold: float

    @classmethod
    def from_parts(cls, atol: float, rtol: float, history_length: int, std_threshold: float):
        return cls(atol=atol, rtol=rtol, history_length=history_length, std_threshold=std_threshold)

    def __call__(
            self,
            data: LocalUpdateResult,
            active_set_pre_filtered: MaskNd,
            active_set_expanded: MaskNd,
            values_next: ArrayNd
    ) -> MaskNd:
        changed = active_set_expanded & ~jnp.isclose(
            data.get_recent_values(), values_next,
            atol=self._atol, rtol=self._rtol
        )

        recent_values_list = data.get_recent_values_list(span=self._history_length)
        recent_values_array = jnp.stack(recent_values_list, axis=-1)
        values_std = jnp.std(recent_values_array, axis=-1)
        values_mean = jnp.mean(recent_values_array, axis=-1)

        where_narrow_std = values_std < self._std_threshold
        where_close_to_mean = jnp.abs(values_next - values_mean) <= values_std
        oscillating = where_narrow_std & where_close_to_mean

        return changed & ~oscillating
