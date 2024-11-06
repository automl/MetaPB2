"""
Modified version of https://github.com/automl/CARL/blob/main/carl/envs/brax/carl_brax_env.py
Avoiding a bug where Carl does not pass the context to the environment.
"""

from __future__ import annotations

import warnings

import brax
import gymnasium
import numpy as np
from jax import numpy as jp

from carl.context.selection import AbstractSelector
from carl.envs.brax.wrappers import GymWrapper, VectorGymWrapper
from carl.envs.carl_env import CARLEnv
from carl.utils.types import Contexts

class CARLBraxEnvFix(CARLEnv):
    env_name: str
    backend: str = "spring"

    def __init__(
            self,
            env: brax.envs.env.Env | None = None,
            batch_size: int = 1,
            contexts: Contexts | None = None,
            obs_context_features: list[str] | None = None,
            obs_context_as_dict: bool = True,
            context_selector: AbstractSelector | type[AbstractSelector] | None = None,
            context_selector_kwargs: dict = None,
            **kwargs,
    ) -> None:
        """
        CARL Gymnasium Environment.

        Parameters
        ----------

        env : brax.envs.env.Env | None
            Brax environment, the default is None.
            If None, instantiate the env with brax' make function and
            `self.env_name` which is defined in each child class.
        batch_size : int
            Number of environments to batch together, by default 1.
        contexts : Contexts | None, optional
            Context set, by default None. If it is None, we build the
            context set with the default context.
        obs_context_features : list[str] | None, optional
            Context features which should be included in the observation, by default None.
            If they are None, add all context features.
        context_selector: AbstractSelector | type[AbstractSelector] | None, optional
            The context selector (class), after each reset selects a new context to use.
             If None, use a round robin selector.
        context_selector_kwargs : dict, optional
            Optional keyword arguments for the context selector, by default None.
            Only used when `context_selector` is not None.

        Attributes
        ----------
        env_name: str
            The registered gymnasium environment name.
        backend: str

        """
        if env is None:
            bs = batch_size if batch_size != 1 else None
            env = brax.envs.create(
                env_name=self.env_name, backend=self.backend, batch_size=bs
            )
            # We have to set the context at this stage because there was a bug that did not allow us to do that later
            warnings.warn('Only the context with id 0 will be picked!')
            if 'gravity' in contexts[0]:
                env.unwrapped.sys = env.unwrapped.sys.replace(gravity=jp.array([0, 0, contexts[0]['gravity']]))
            # Brax uses gym instead of gymnasium
            if batch_size == 1:
                env = GymWrapper(env)
            else:
                env = VectorGymWrapper(env)

            # The observation space also needs to from gymnasium
            env.observation_space = gymnasium.spaces.Box(
                low=env.observation_space.low,
                high=env.observation_space.high,
                dtype=np.float32,
            )

        super().__init__(
            env=env,
            contexts=contexts,
            obs_context_features=obs_context_features,
            obs_context_as_dict=obs_context_as_dict,
            context_selector=context_selector,
            context_selector_kwargs=context_selector_kwargs,
            **kwargs,
        )

    def _update_context(self) -> None:
        pass
