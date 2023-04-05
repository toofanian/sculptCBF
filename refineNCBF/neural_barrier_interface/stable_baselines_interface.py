from typing import Callable, Union

import attr
import stable_baselines3
import numpy as np


@attr.s(auto_attribs=True)
class StableBaselinesCallable(Callable):
    network: Union[stable_baselines3.PPO, stable_baselines3.SAC]

    def __call__(self, state):
        try:
            return self.network.predict(state, deterministic=True)[0]
        except ValueError:
            return (self.network.action_space.low + self.network.action_space.high) / 2
