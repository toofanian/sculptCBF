from abc import ABC, abstractmethod
from typing import Tuple, Optional

import attr
from jax import numpy as jnp

from refineNCBF.utils.types import ArrayNd, MaskNd


@attr.s(auto_attribs=True, eq=False)
class ValuePostprocessor(ABC):
    @abstractmethod
    def __call__(self, t, x):
        ...


@attr.s(auto_attribs=True, eq=False)
class ValuePostprocessorSequence(ValuePostprocessor):
    value_postprocessors: Tuple[ValuePostprocessor, ...]

    @classmethod
    def from_value_postprocessors(cls, value_postprocessors: Tuple[ValuePostprocessor, ...]):
        return cls(value_postprocessors)

    def __call__(self, t, x):
        for value_postprocessor in self.value_postprocessors:
            x = value_postprocessor(t, x)
        return x


@attr.s(auto_attribs=True, eq=False)
class NotBiggerator(ValuePostprocessor):
    values: ArrayNd
    enforcement_region: MaskNd

    @classmethod
    def from_array(cls, values: ArrayNd, enforcement_region: Optional[MaskNd] = None) -> 'NotBiggerator':
        if enforcement_region is None:
            enforcement_region = jnp.zeros_like(values, dtype=bool)
        values = jnp.array(values)
        return cls(values=values, enforcement_region=enforcement_region)

    def __call__(self, t, x):
        return jnp.where(self.enforcement_region, jnp.minimum(x, self.values), self.values)


@attr.s(auto_attribs=True, eq=False)
class NotSmallerator(ValuePostprocessor):
    values: ArrayNd
    enforcement_region: MaskNd

    @classmethod
    def from_array(cls, values: ArrayNd, enforcement_region: Optional[MaskNd] = None) -> 'NotSmallerator':
        if enforcement_region is None:
            enforcement_region = jnp.zeros_like(values, dtype=bool)
        values = jnp.array(values)
        return cls(values=values, enforcement_region=enforcement_region)

    def __call__(self, t, x):
        return jnp.where(self.enforcement_region, jnp.maximum(x, self.values), self.values)


@attr.s(auto_attribs=True, eq=False)
class Freezerator(ValuePostprocessor):
    """
    Freezes the value of the state at the terminal time. Used to test convergence of the value function.
    """
    values: ArrayNd
    enforcement_region: MaskNd

    @classmethod
    def from_array(cls, values: ArrayNd, enforcement_region: Optional[MaskNd] = None) -> 'Freezerator':
        if enforcement_region is None:
            enforcement_region = jnp.zeros_like(values, dtype=bool)

        return cls(values=values, enforcement_region=enforcement_region)

    def __call__(self, t, x):
        return jnp.where(self.enforcement_region, self.values, x)


@attr.s(auto_attribs=True, eq=False)
class ReachAvoid(ValuePostprocessor):
    values: ArrayNd
    reach_set: MaskNd

    @classmethod
    def from_array(cls, values: ArrayNd, reach_set: Optional[MaskNd] = None) -> 'ReachAvoid':
        if reach_set is None:
            reach_set = jnp.zeros_like(values, dtype=bool)
        return cls(values=values, reach_set=reach_set)

    def __call__(self, t, x):
        v_enforce_obstacle = jnp.minimum(x, self.values)
        v_enforce_safety = jnp.where(self.reach_set, self.values, v_enforce_obstacle)
        return v_enforce_safety
