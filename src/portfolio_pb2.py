import argparse
import importlib
import json
import os
import sys
import time
from ray.tune import sample_from

from src.config_processor import process_config_object, process_configs
from src.algorithms.algorithm_utils import sampler
from src.run_experiment import save_experiment, save_schedules
from src.utils import execute_experiment
from src.utils import process_result_grid

def warm_start_config(path_to_experiment, portfolio_save_dir, run_idx):
    experiment_config = process_configs(path_to_experiment, run_idx)
    dataset = str(json.loads(experiment_config['trial_name'])['task_name'])
    portfolio_path = portfolio_save_dir + f"/{dataset}_portfolio.json"
    portfolio = read_portfolio(portfolio_path) # get the name stuff
    for hp in experiment_config['hyperparameters']:
        experiment_config['param_space'][hp] = sample_from(sampler(portfolio, hp))
    return experiment_config

def read_portfolio(portfolio_path):
    with open(portfolio_path, 'r') as file:
        portfolio = json.load(file)
    return portfolio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs experiment defined in configuration file with warmstarting')
    parser.add_argument('--moab_id', default='0', type=int,
                        help="Specifies which experiment configuration to run")
    parser.add_argument('--config_file_name', default='mountaincar_discrete_config', type=str,
                        help='Name of the configuration file of the experiment')
    # parser.add_argument('--portfolio_save_dir', default='ray_results/mountaincar_discrete_config_2023-05-26/bo/carl_mountaincar_discrete./initial_configs/portfolio', type=str,
    #                     help='where the initial configs are saved - used to get ')
    parser.add_argument('--portfolio_save_dir', default='ray_results/cartpole_config_2023-06-04/portfolio/', type=str,
                        help='where the initial configs are saved - used to get ')
    parser.add_argument('--experiment_dir', default='CARL_experiment_configurations', type=str, help='')
    parser.add_argument('--save_meta', action="store_true", default=False,
                        help="Savees data for later use in metaPB2 methods.")
    
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    portfolio_save_dir = args.portfolio_save_dir
    moab_id = args.moab_id

    warm_started_configuration = warm_start_config(experiment_dir + '/' + args.config_file_name,
                                                          portfolio_save_dir, moab_id)
    t0 = time.time()
    result_grid = execute_experiment(warm_started_configuration)
    t1 = time.time()
    results_metrics, warm_started_configuration, errors = process_result_grid(result_grid, warm_started_configuration)
    results_metrics['total_run_time'] = t1 - t0
    if args.save_meta:
            save_schedules(warm_started_configuration, args.config_file_name)
    save_experiment(results_metrics, warm_started_configuration, errors)
