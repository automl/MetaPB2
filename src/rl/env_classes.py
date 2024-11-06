import logging

import gymnasium as gym
import numpy as np
from carl.envs import CARLMountainCar, CARLCartPole, CARLPendulum, CARLAcrobot
from carl.envs.gymnasium.carl_gymnasium_env import CARLGymnasiumEnv
from etils import epath
from gymnasium import Space
from gymnasium.wrappers import TimeLimit
from matplotlib import pyplot as plt
from ray.tune.registry import register_env
import random
import torch

from src.rl.brax_env.carl_halfcheetah_fix import CARLBraxHalfcheetahFix
from src.rl.brax_env.carl_hopper_fix import CARLBraxHopperFix
from src.rl.brax_env.carl_humanoid_fix import CARLBraxHumanoidFix
from src.rl.brax_env.carl_humanoidstandup import CARLBraxHumanoidStandupFix
from src.rl.brax_env.carl_inverted_double_pendulum_fix import CARLBraxInvertedDoublePendulumFix
from src.rl.brax_env.carl_inverted_pendulum_fix import CARLBraxInvertedPendulumFix
from src.rl.brax_env.carl_pusher_fix import CARLBraxPusherFix
from src.rl.brax_env.carl_reacher_fix import CARLBraxReacherFix
from src.rl.brax_env.carl_walker2d import CARLBraxWalker2dFix


def get_carl_class(env_name):
    if env_name == 'humanoid':
        carl_class = CARLBraxHumanoidFix
    elif env_name == 'halfcheetah':
        carl_class = CARLBraxHalfcheetahFix
    elif env_name == 'hopper':
        carl_class = CARLBraxHopperFix
    elif env_name == 'humanoid_standup':
        carl_class = CARLBraxHumanoidStandupFix
    elif env_name == 'inverted_double_pendulum':
        carl_class = CARLBraxInvertedDoublePendulumFix
    elif env_name == 'inverted_pendulum':
        carl_class = CARLBraxInvertedPendulumFix
    elif env_name == 'pusher':
        carl_class = CARLBraxPusherFix
    elif env_name == 'reacher':
        carl_class = CARLBraxReacherFix
    elif env_name == 'walker2d':
        carl_class = CARLBraxWalker2dFix
    elif env_name == 'mountain_car':
        carl_class = CARLMountainCar
    elif env_name == 'cart_pole':
        carl_class = CARLCartPole
    elif env_name == 'pendulum':
        carl_class = CARLPendulum
    elif env_name == 'acrobot':
        carl_class = CARLAcrobot
    else:
        logging.info(f"""'{env_name}' is not one of the supported carl environments. 
        It is assumed that the environment is already registered.""")
        return None

    class MyEnv(gym.Env):
        def __init__(self, env_config=None):
            if env_config is None:
                env_config = {}

            context = carl_class.get_default_context()
            if 'gravity' in env_config:
                gravity = env_config['gravity']
                if 'gravity' in context:
                    context['gravity'] = gravity
            elif 'g_factor' in env_config:
                g_factor = env_config['g_factor']
                if 'gravity' in context:
                    context['gravity'] = context['gravity'] * g_factor
                elif 'LINK_MASS_1' in context and 'LINK_MASS_2' in context:
                    # scale the mass by g factor to simulate increased gravity
                    context['LINK_MASS_1'] = context['LINK_MASS_1'] * g_factor
                    context['LINK_MASS_2'] = context['LINK_MASS_2'] * g_factor


            self.wrapped_env = carl_class(hide_context=True, contexts={0: context}) # hide context has no affect?
            a_space = self.wrapped_env.action_space
            p_space = self.wrapped_env.observation_space.spaces['obs']

            self.is_gymnasium = issubclass(carl_class, CARLGymnasiumEnv)
            if self.is_gymnasium:
                self.action_space = a_space
                self.observation_space = p_space
            else:
                self.action_space = gym.spaces.Box(low=a_space.low, high=a_space.high, shape=a_space.shape,
                                                   dtype=a_space.dtype, seed=env_config.get('seed', None))
                self.observation_space = gym.spaces.Box(low=p_space.low, high=p_space.high, shape=p_space.shape,
                                                        dtype=p_space.dtype, seed=env_config.get('seed', None))
            self.render_mode = self.wrapped_env.render_mode

        def reset(self, *args, seed=None, options=None):
            observation, info = self.wrapped_env.reset(*args, seed=seed, options=options)
            if self.is_gymnasium:
                return observation['obs'], info
            else:
                return observation['obs']._value, info

        def step(self, action):
            observation, reward, terminated, truncated, info = self.wrapped_env.step(np.array(action))
            if self.is_gymnasium:
                if not np.isfinite(observation['obs']).all() or not np.isfinite(reward).all():
                    print('Env error')
                return observation['obs'], reward, terminated, truncated, info
            else:
                if not np.isfinite(observation['obs']._value).all() or not np.isfinite(float(reward)).all():
                    print('Env error')
                return observation['obs']._value, float(reward), bool(terminated), truncated, info

        def render(self):
            return self.wrapped_env.render()
    return MyEnv


def register_carl_env(env_name):
    MyEnv = get_carl_class(env_name)

    if MyEnv is None:
        return

    def env_creator(env_config):
        if 'max_episode_steps' in env_config.keys():
            max_episode_steps = env_config.pop('max_episode_steps')
            return TimeLimit(MyEnv(env_config), max_episode_steps=max_episode_steps)
        else:
            return MyEnv(env_config)

    register_env(env_name, env_creator)

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    import imageio
    set_seeds(3)
    class_carl = CARLBraxPusherFix
    context = class_carl.get_default_context()
    gravity = 0.000001

    context['gravity'] = -gravity

    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)


    env = class_carl(contexts={0: context})
    path = epath.resource_path("brax") / env.asset_path

    env = class_carl(contexts={0: context})

    stopped = False
    env.reset(seed=3)
    print(env.context)

    frames = []
    for _ in range(500):
        random.seed(3)
        np.random.seed(3)
        torch.manual_seed(3)
        action=np.ones_like(env.action_space.sample())
        # print(action)
        observation, reward, terminated, truncated, info = env.step(action)
        stopped = terminated or truncated
        frames.append(env.render())
        # env.reset()
    print(reward)
    print(env.context)
    imageio.mimsave('vid.gif', frames, duration=0.1)