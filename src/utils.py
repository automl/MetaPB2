import json
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Callable

import imageio
import numpy as np
import pandas as pd
import torch
from ray import tune, air
from ray.rllib.core.rl_module import RLModule

from src.config_processor import process_configs
from src.rl.env_classes import get_carl_class


def get_src_path() -> Path:
    return Path(__file__).absolute().parent


def run_configuration(path_to_experiment, run_idx):
    experiment_config = process_configs(path_to_experiment, run_idx)
    # Create object, maybe do some finessing
    t0 = time.time()
    result_grid = execute_experiment(experiment_config)
    t1 = time.time()
    visualize_agent(result_grid.get_best_result(), experiment_config, run_idx)
    results_metrics, experiment_config, errors = process_result_grid(result_grid, experiment_config)
    results_metrics['total_run_time'] = t1 - t0
    return results_metrics, experiment_config, errors


def visualize_agent(best_config, experiment_config, run_idx):
    best_checkpoint = best_config.checkpoint  # get the checkpoint and load an agent from the checkpoint
    rl_module = RLModule.from_checkpoint(
        Path(best_checkpoint.path) / "learner" / "module_state" / "default_policy"
        )

    # initialize environment and add rendering to the environments.
    env_name = experiment_config['param_space']['env']
    env_config = experiment_config['param_space']['env_config']
    env = get_carl_class(env_name)(env_config)

    frames = []
    episode_return = 0
    terminated = truncated = False

    obs, info = env.reset()

    while not terminated and not truncated:
        # Compute the next action from a batch (B=1) of observations.
        torch_obs_batch = torch.from_numpy(np.array([obs]))
        action_logits = rl_module.forward_inference({"obs": torch_obs_batch})[
            "action_dist_inputs"
        ]
        # The default RLModule used here produces action logits (from which
        # we'll have to sample an action or use the max-likelihood one).
        action = torch.argmax(action_logits[0]).numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        frames.append(env.render())

    print(f"Reached episode return of {episode_return}.")
    # I need the save directory to save the model to

    save_dir = Path(experiment_config['run_config']['local_dir']).parent
    imageio.mimsave(save_dir / f'{run_idx}-{env_name}-{env_config["name"]}.gif', frames, fps=30)


def execute_experiment(experiment_config):
    base_dir = "./ray_results/"
    experiment_config['run_config']['local_dir'] = str(
        Path(base_dir + experiment_config['run_config']['local_dir'] + '/tmp').absolute())
    # experiment_config['run_config']['name'] += '/tmp'

    run_config = air.RunConfig(**experiment_config['run_config'])
    tune_config = tune.TuneConfig(**experiment_config['tune_config'])
    param_space = experiment_config['param_space']
    seed = experiment_config['seed']
    # set seeds for numpy, random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trainable_name = experiment_config["algorithm"]

    tuner = tune.Tuner(
        trainable_name,
        run_config=run_config,  # Done!
        tune_config=tune_config,  # Done
        param_space=param_space,
    )
    result_grid = tuner.fit()
    return result_grid


def process_result_grid(result_grid, experiment_config):
    results_metrics = []
    for result in result_grid:
        results_metrics.append(result.metrics_dataframe.rename(columns=lambda x: x.split('/')[-1]))
    results_metrics = pd.concat(results_metrics, axis=0).reset_index().rename(columns={"index": "iteration"})
    errors = result_grid.errors
    return results_metrics, experiment_config, errors


def get_experiment_type(config):
    # tells if the experiment is DL or RL
    if hasattr(config, "combined_config"):
        algorithm = config.combined_config["algorithm"]["algorithm"][0]
    else:
        algorithm = config["algorithm"]

    if isinstance(algorithm, str):
        return "rl"
    else:
        return "dl"


def get_hyperparam_bounds(hyperparam_bounds_path="hyperparam_info/hyperparam_bounds.json"):
    """
    takes a json file path that contains {"hp":[min,max]}

    Returns: dict of hyperparam bounds defined in json file

    """
    # this return statement is a temporary fix because of weird file system issues that happen on nemo
    return {"lr": [0.000001, 0.02], "weight_decay": [1e-8, 1e-3],
            "attention_dropout": [0., 0.5],
            "ffn_dropout": [0., 0.5]}
    absolute_path = os.path.abspath(hyperparam_bounds_path)
    with open(absolute_path) as file:
        data = json.load(file)
    return data


def rl_metric_rename(results_metrics, metric):
    metric_duplicates = results_metrics[metric.split('/')[-1]]
    idx_missing = np.argmax(metric_duplicates.isnull().any().values)
    results_metrics[metric] = metric_duplicates.iloc[:, idx_missing]
    return results_metrics


def get_keep_bounds(hpyerparam_bounds: Dict[str, List[float]]) -> Callable:
    def keep_bounds(old_config):
        for key, value in hpyerparam_bounds.items():
            if key in old_config:
                old_config_value = old_config[key]
                old_config[key] = min(max(old_config_value, value[0]), value[1])
        return old_config

    return keep_bounds


def get_float_transformer(hyperparameters):
    def float_config_transformer(old_config):
        for hyperparameter in hyperparameters:
            old_config[hyperparameter] = float(old_config[hyperparameter])
        return old_config

    return float_config_transformer
