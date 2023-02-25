from typing import Callable, Union

import attr
import stable_baselines3


@attr.s(auto_attribs=True)
class StableBaselinesCallable(Callable):
    network: Union[stable_baselines3.PPO, stable_baselines3.SAC]

    def __call__(self, state):
        return self.network.predict(state, deterministic=True)[0]
