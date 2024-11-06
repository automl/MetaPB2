import json
from collections import OrderedDict

from ray import tune
from ray.air import FailureConfig

from src.rl.environment_families import FAMILIES, PLANET_GRAVITY, ABSOLUTE_PLANET_GRAVITY

num_seeds = 5
seed = OrderedDict({'seed': [i for i in range(num_seeds)]})

"""
Specifying the run environment
"""


def get_trial_name(experiment_config):
    algorithm = experiment_config['tune_config']['scheduler']['type']
    env = experiment_config['param_space']["env"]
    env_config = experiment_config['param_space']["env_config"]
    if 'name' in env_config:
        env += f" {env_config['name']}"
    trial_name = {'task_name': env, 'algorithm': f'{algorithm}'}
    return json.dumps(trial_name)

"""
Specifying the parameter space
"""
param_space = OrderedDict({
    "seed": [tune.randint(0, 10_000)],
    "framework": ["torch"],
    "lr": [tune.loguniform(1e-5, 1e-3)],
    "lambda": [tune.uniform(0.9, 0.99)],
    "clip_param": [tune.uniform(0.1, 0.5)],
    "train_batch_size": [25_000],
    "env": FAMILIES["classic_control"],
    "env_config": [{'g_factor': 1}],
    "num_workers": [0],
    "recreate_failed_workers": [True],
    "max_num_worker_restarts": [20],
    "delay_between_worker_restarts_s": [30],
    "evaluation_interval": [4],
    "always_attach_evaluation_results": [True],
})


"""
Algorithm Configs
"""

algorithm = OrderedDict({"algorithm": ['PPO']})

"""
tune_config

"""

scheduler_config_log = OrderedDict({
    "time_attr": "timesteps_total",
    "perturbation_interval": 100_000,
    "hyperparam_bounds": {
            "lambda": [0.9, 0.99],
            "clip_param": [0.1, 0.5],
            "lr": [1e-5, 1e-3],
    },
    # "log_scale_hyperparam": [],
    "synch": True,
    # "use_hp_initializer": True,
})

tune_config = OrderedDict({
    "metric": ["evaluation/sampler_results/episode_reward_mean"],
    "mode": ["max"],
    "scheduler": [
        dict(type='PB2', **scheduler_config_log),
    ],
    "num_samples": [4],
})

"""
Runconfig
"""
run_config = OrderedDict({
    "stop": [{"timesteps_total": 1_500_000}],
    "verbose": [4],
    "log_to_file": [True],
    "failure_config": [FailureConfig(
        max_failures=1
    )]
})

hyperparameters = ["lr", "lambda", "clip_param", "train_batch_size"]

combined_config = OrderedDict({
    "seed": seed,
    "param_space": param_space,
    "algorithm": algorithm,
    "tune_config": tune_config,
    "run_config": run_config,
})
