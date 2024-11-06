"""
Second step of the portfolio algorithm.
Reruns the best configurations of each tasks on all other tasks to later construct a performance matrix.
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from types import ModuleType

from ray.tune.search import BasicVariantGenerator

from src.config_processor import configure_experiment
from src.run_experiment import clean_up_run
from src.utils import get_experiment_type, rl_metric_rename, process_result_grid, execute_experiment, get_src_path


def generate_rerun_configurations(initial_regions_array: list, config_file_name: str) -> ModuleType:
    """
    Generates a configuration object for rerunning the best hyperparameter configurations.
    Args:
        initial_regions_array (list): A list of dictionaries where each dictionary represents a hyperparameter configuration.
        config_file_name (str): Name of the configuration file of the experiment.

    Returns:
        module: A configuration module with updated initial regions.
    """
    # the algo type is inferred from config file, but can alter it if the object has a set method  to change the HPs
    # initial regions dict format: {"hyperparam1" : [c1, c2, c3.....], "hyperparam2" : [c1,c2...]}
    config_file_path = os.path.dirname(str(get_src_path().parent) + "/" + config_file_name)
    sys.path.insert(1, config_file_path)  # #hack
    import importlib
    config = importlib.import_module(config_file_name.split("/")[-1], package=None)
    config.combined_config["initial_regions_array"] = OrderedDict({'initial_regions_array': initial_regions_array})

    return config


def read_initial_regions(initial_configs_save_dir:str) -> list:
    """
    Reads the initial best hyperparameter configurations from the initial_configs_save_dir and returns
    them as a list of dicts.
    Args:
        initial_configs_save_dir (str): Path to the directory containing the initial configuration files.

    Returns:
        list: A list of dictionaries where each dictionary represents a hyperparameter configuration.
              The dictionary format is: {"score": value, "hyperparam1": value, "hyperparam2": value, ...}
    """
    file_extension = ".json"
    initial_regions_array = []

    for filename in os.listdir(initial_configs_save_dir):
        if filename.endswith(file_extension):
            file_path = os.path.join(initial_configs_save_dir, filename)
            with open(file_path, 'r') as file:
                # initial region hps are in form of a dict {score: value , <hp_name>: hp_value}
                initial_region_hps = json.load(file)
                if isinstance(initial_region_hps, list):
                    # this happens if we decide to rerun only elements contstructed from
                    # an already built portfolio
                    initial_regions_array.extend(initial_region_hps)
                else:
                    initial_regions_array.append(initial_region_hps)
    return initial_regions_array


def rerun_configurations(config_file_name: str, initial_configs_save_dir: str, moab_id: int) -> None:
    """
    Runs the experiment defined in the configuration file and returns the results.

    Args:
        config_file_name (str): Name of the configuration file of the experiment.
        initial_configs_save_dir (str): Path to the directory containing the initial configuration files.
        moab_id (int): Specifies which hyperparameter configuration to rerun on which environment.

    Returns:
        None
    """
    initial_regions_array = read_initial_regions(initial_configs_save_dir)
    config = generate_rerun_configurations(initial_regions_array, config_file_name)
    experiment_config = configure_experiment(config, moab_id, config_file_name)

    # set constant hyperparameters
    hp_config = experiment_config["initial_regions_array"]["initial_regions_array"]
    del hp_config['score']
    experiment_config['param_space'].update(hp_config)
    # set the random sampling to zero
    experiment_config["tune_config"]['num_samples'] = 1

    search_object = BasicVariantGenerator(random_state=0, constant_grid_search=True)
    experiment_config["tune_config"].update({"search_alg": search_object})

    result_grid = execute_experiment(experiment_config)

    results_metrics, _, _ = process_result_grid(result_grid, experiment_config)

    experiment_type = get_experiment_type(config)
    metric = experiment_config["tune_config"]["metric"]
    if experiment_type == "rl":  # RL results are weird. We are assuming that eval values have missing entries
        results_metrics = rl_metric_rename(results_metrics, metric)

    results_last_iteration = results_metrics[results_metrics['iteration'] == results_metrics['iteration'].max()]
    hp_config = results_last_iteration.groupby(by=experiment_config["hyperparameters"]).agg(
        {metric: ['mean', 'std']}).reset_index()
    score = hp_config[metric]['mean'].iloc[0]
    score_std = hp_config[metric]['std'].iloc[0]
    hp_config.drop(columns=[metric], inplace=True)
    hp_config = hp_config.droplevel(1, axis=1).iloc[0].to_dict()

    result = {'score': score, 'score_std': score_std, 'run_env': experiment_config['trial_name'],
              'source_env': experiment_config['param_space']['source_env'].removesuffix('.csv'),
              'hyperparameters': hp_config}

    # save results
    save_dir = Path(initial_configs_save_dir).parent / 'rerun_configs' / f'{moab_id}.json'
    save_dir.parent.mkdir(exist_ok=True)
    with open(save_dir, "w") as f:
        json.dump(result, f)

    # save stats
    num_settings = len(initial_regions_array)

    num_envs = len(config.combined_config['param_space']["env"]) * len(
        config.combined_config['param_space']["env_config"])


    mode = experiment_config["tune_config"]["mode"].lower()
    assert mode in ['min', 'max'], "Invalid mode specified"

    # stats dict is needed to construct the porfolio matrix dimensions
    stats_dict = {"num_settings": num_settings, "num_envs": num_envs, "mode": mode}  # todo: remove num_seeds
    # saved under env dir
    dir = Path(initial_configs_save_dir).parent / 'stats.json'
    dir.parent.mkdir(exist_ok=True)
    with open(dir, 'w') as f:
        json.dump(stats_dict, f)

    local_dir = experiment_config["run_config"]["local_dir"]
    run_name = experiment_config["run_config"]["name"]
    clean_up_run(local_dir, run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs experiment defined in configuration file.')
    parser.add_argument('--moab_id', default=0, type=int,
                        help="Specifies which hyperparameter configuration to rerun on which environment.")
    parser.add_argument('--config_file_name', type=str,
                        help='Name of the configuration file of the experiment')
    parser.add_argument('--initial_configs_save_dir', type=str,
                        help='Path to the experiment directory that contains the run configuration file.')
    parser.add_argument('--experiment_dir', type=str,
                        help='Relative path to the experiment directory that contains the run configuration file '
                             'starting from the root directory.')

    args = parser.parse_args()
    experiment_dir = args.experiment_dir

    rerun_configurations(
        config_file_name=experiment_dir + '/' + args.config_file_name,
        initial_configs_save_dir=str(args.initial_configs_save_dir),
        moab_id=int(args.moab_id))
