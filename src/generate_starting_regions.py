"""
First step of the portfolio algorithm.
Finds the best hyperparameters for all tasks and saves them in a directory for later use.
"""

import argparse
import json
import os
import sys

import pandas as pd

from src.config_processor import configure_experiment
from src.run_experiment import clean_up_run
from src.utils import execute_experiment, get_experiment_type, process_result_grid, rl_metric_rename, get_src_path
from ray.tune.search import BasicVariantGenerator


def run_generate_configs(config_file: str, moab_id: int, max_concurrent: int = 4) -> (pd.DataFrame, dict):
    """
    Runs the experiment defined in the configuration file and returns the results.

    Args:
        config_file (str): The path to the configuration file that defines the experiment.
        moab_id (int): The ID of the experiment configuration to run.
        max_concurrent (int, optional): The maximum number of concurrent trials. Defaults to 4.

    Returns:
        pd.DataFrame: A DataFrame containing the results metrics of the experiment.
        dict: A dictionary containing the experiment configuration.
    """
    config_file_path = os.path.dirname(str(get_src_path().parent) + "/" + config_file)
    sys.path.insert(1, config_file_path)  # hack
    import importlib

    config = importlib.import_module(config_file.split("/")[-1], package=None)
    experiment_type = get_experiment_type(config)

    experiment_config = configure_experiment(config, moab_id, config_file)
    search_object = BasicVariantGenerator(random_state=moab_id, constant_grid_search=True,
                                          max_concurrent=max_concurrent)

    experiment_config["tune_config"].update({"search_alg": search_object})

    print(experiment_config)
    result_grid = execute_experiment(experiment_config)
    results_metrics, _, _ = process_result_grid(result_grid, experiment_config)

    if experiment_type == "rl":
        metric = experiment_config["tune_config"]["metric"]
        results_metrics = rl_metric_rename(results_metrics, metric)
    return results_metrics, experiment_config


def save_best_config(results_metrics: pd.DataFrame, experiment_config: dict, n_best: int) -> None:
    """
    Saves the best n configurations as json files in the initial_configs directory.

    Args:
        results_metrics (pd.DataFrame): DataFrame containing the results metrics of the experiment.
        experiment_config (dict): Dictionary containing the experiment configuration.
        n_best (int): The number of top configurations to save.

    Returns:
        None
    """
    metric = experiment_config["tune_config"]["metric"]
    results_last_iteration = results_metrics[results_metrics['iteration'] == results_metrics['iteration'].max()]
    scores = results_last_iteration.groupby(by=experiment_config["hyperparameters"]).agg({metric: 'mean'}).reset_index()
    scores['source_env'] = experiment_config['trial_name']

    if experiment_config['tune_config']['mode'] == 'max':
        best_configs = scores.nlargest(n_best, metric)
    else:
        best_configs = scores.nsmallest(n_best, metric)
    best_configs.rename(columns={metric: 'score'}, inplace=True)

    local_dir = experiment_config["run_config"]["local_dir"]
    save_dir = '/'.join(local_dir.split('/')[:-1]) + "/initial_configs/"

    if not (os.path.exists(save_dir)):
        os.makedirs(save_dir)

    for i in range(n_best):
        current_config = best_configs.iloc[i].to_dict()
        file_name = f"{experiment_config['trial_name']}_{i}.json"
        with open(save_dir + "/" + file_name, "w+") as fp:
            json.dump(current_config, fp)

    run_name = experiment_config["run_config"]["name"]
    clean_up_run(local_dir, run_name)


def save_results(results_metrics: pd.DataFrame, experiment_config: dict) -> None:
    """
    Saves the results in the ray_results directory. The directory structure matches the config file directory structure.

    Args:
        results_metrics (pd.DataFrame): DataFrame containing the results metrics of the experiment.
        experiment_config (dict): Dictionary containing the experiment configuration.

    Returns:
        None
    """
    local_dir = experiment_config["run_config"]["local_dir"]
    save_dir = '/'.join(local_dir.split('/')[:-1]) + "/initial_results/"
    if not (os.path.exists(save_dir)):
        os.makedirs(save_dir)
    results_metrics.to_csv(save_dir + f"{experiment_config['trial_name']}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs experiment defined in configuration file.')
    parser.add_argument('--moab_id', default=0, type=int,
                        help="Specifies which experiment configuration to run")
    parser.add_argument('--config_file_name', type=str,
                        help='Name of the configuration file of the experiment')
    parser.add_argument('--experiment_dir', type=str,
                        help='Relative path to the experiment directory that contains the run configuration file '
                             'starting from the root directory.')
    parser.add_argument('--max_concurrent', default=1, type=int,
                        help="Maximum concurrent trials. This should match the number of available cpu's for maximal efficiency.")
    parser.add_argument('--n_best_configs', default=1, type=int,
                        help='The top n configurations that are returned as candidates.')
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    results_metrics, experiment_config = run_generate_configs(experiment_dir + '/' + args.config_file_name,
                                                              int(args.moab_id),
                                                              int(args.max_concurrent))

    save_best_config(results_metrics, experiment_config, args.n_best_configs)
    save_results(results_metrics, experiment_config)
