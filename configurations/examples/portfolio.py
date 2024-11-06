import json
from collections import OrderedDict

from ray import tune
from ray.air import FailureConfig

from src.rl.environment_families import FAMILIES, G_PLANET

num_seeds = 1 # Ignored
seed = OrderedDict({'seed': [i for i in range(num_seeds)]})

"""
Specifying the run environment
"""


def get_trial_name(experiment_config):
    env = experiment_config['param_space']["env"]
    env_config = experiment_config['param_space']["env_config"]
    if 'name' in env_config:
        env += f" {env_config['name']}"
    return env

"""
Specifying the parameter space
"""
param_space = OrderedDict({
    "seed": [tune.grid_search([0, 1])], # Number of seeds that are getting averaged over for scores
    "framework": ["torch"],
    "lr": [tune.loguniform(1e-5, 1e-3)],
    "lambda": [tune.uniform(0.9, 0.99)],
    "clip_param": [tune.uniform(0.1, 0.5)],
    "train_batch_size": [1_000],
    "env": ['mountain_car', 'cart_pole',],
    "env_config": [
        {'g_factor': 9.81/9.81, 'name': 'earth'},
        {'g_factor': 11./9.81, 'name': 'neptune'},
    ],
    "num_workers": [0],
    "recreate_failed_workers": [True],
    "max_num_worker_restarts": [20],
    "delay_between_worker_restarts_s": [30],
    "evaluation_interval": [2],
    "always_attach_evaluation_results": [True],
})


"""
Algorithm Configs
"""

algorithm = OrderedDict({"algorithm": ['PPO']})

"""
tune_config

"""

tune_config = OrderedDict({
    "metric": ["evaluation/sampler_results/episode_reward_mean"],
    "mode": ["max"],
    "num_samples": [4], # Total number of trials to run initially
})

"""
Runconfig
"""
run_config = OrderedDict({
    "stop": [{"timesteps_total": 2_000}],
    "verbose": [4],
    "log_to_file": [True],
    "failure_config": [FailureConfig(
        max_failures=1
    )]
})

hyperparameters = ["lr", "lambda", "clip_param"]

combined_config = OrderedDict({
    "seed": seed,
    "param_space": param_space,
    "algorithm": algorithm,
    "tune_config": tune_config,
    "run_config": run_config,
})
