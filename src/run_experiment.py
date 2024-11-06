import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from src import utils
from src.algorithms.algorithm_utils import process_data
from src.algorithms.log_pb2 import pb2_gp
from src.algorithms.meta_data import transform_meta_data
def save_experiment(results_metrics, experiment_config, errors):
    local_dir = experiment_config['run_config']['local_dir']
    name = experiment_config['run_config']['name'].split('_')[0]
    save_dir = '/'.join(local_dir.split('/')[:-1])
    if not (os.path.exists(save_dir)):
        os.makedirs(save_dir)

    # documenting errors
    if errors is not None and len(errors) > 0:
        with open(save_dir + f"/{name}_errors.json", 'w') as f:
            json.dump([str(e) for e in errors], f)
    if len(results_metrics) == 0:
        return

    hyperparameters = experiment_config["hyperparameters"]
    results_metrics['trial_name'] = experiment_config['trial_name']
    results_metrics['seed'] = experiment_config['seed']
    results_metrics.to_csv(save_dir + f"/{name}_result.csv", index=False)

    # save weights if meta method is used
    if 'scheduler' in experiment_config['tune_config']:
        scheduler = experiment_config['tune_config']['scheduler']
        if hasattr(scheduler, 'meta_data'):
            weights = scheduler.meta_data.weights_over_time
            weights['seed'] = experiment_config['seed']
            weights['trial_name'] = experiment_config['trial_name']
            weights.to_csv(save_dir + f"/{name}_weights.csv", index=False)

        if hasattr(scheduler, 'gp_statistics'):
            if 'kernel_parameters' in scheduler.gp_statistics:
                scheduler.gp_statistics['kernel_parameters']['seed'] = experiment_config['seed']
                scheduler.gp_statistics['kernel_parameters']['trial_name'] = experiment_config['trial_name']
                scheduler.gp_statistics['kernel_parameters'].to_csv(save_dir + f'/{name}_kern_param.csv', index=False)

    # Create Info File
    if not (os.path.isfile(save_dir + "/info.json")):
        with open(save_dir + "/info.json", 'w') as f:
            info_dict = {"hyperparameters": hyperparameters}
            json_object = json.dumps(info_dict)
            f.write(json_object)

    # Delete the temporary directory where ray tune run is saved

    # shutil.rmtree(local_dir + '/' + name)  # delete the logged files
    run_name = experiment_config['run_config']['name'] # Ray saves tuner in wrong dir so it is seperately deleted
    clean_up_run(local_dir, run_name)


def clean_up_run(local_dir: str, run_name: str):
    if os.path.exists('./ray_results/' + run_name) and os.path.isdir('./ray_results/' + run_name):
        shutil.rmtree('./ray_results/' + run_name)

    if os.path.exists(local_dir + '/' + run_name) and os.path.isdir(local_dir + '/' + run_name):
        shutil.rmtree(local_dir + '/' + run_name)

def save_schedules(experiment_config, config_name):
    """
    Saves the schedules used for meta methods.
    :param experiment_config:
    :return:
    """
    data = experiment_config['tune_config']['scheduler'].data
    bounds = experiment_config['tune_config']['scheduler']._hyperparam_bounds
    schedule = process_data(data, bounds)
    config_dir = experiment_config['run_config']['local_dir'].split('/')[-3]
    # trial_name should be a dict turned into a string containing the keys algorithm and task_name.
    task_info = json.loads(experiment_config['trial_name'])
    save_dir = f"meta_data/{config_dir}/{config_name}/{experiment_config['seed']}"
    if not (os.path.exists(save_dir)):
        os.makedirs(save_dir)
    schedule.to_csv(f"{save_dir}/{task_info['task_name']}.csv", index=False)


def save_gaussian_process(experiment_config):
    """
    Loads the saved schedule uses it to compute the surrogate model and saves a pickled version of it.
    The GP puts the same hyperparameters on a logscale that the original scheduler did.
    This should match the logscaling used by the metaPB2 algorithm.
    :param experiment_config:
    :return:
    """
    # load schedule
    config_name = experiment_config['config_file_name']
    # trial_name should be a dict turned into a string containing the keys algorithm and task_name.
    task_info = json.loads(experiment_config['trial_name'])
    save_dir = f"meta-data/{config_name}/{task_info['algorithm']}/{experiment_config['seed']}"
    schedule = pd.read_csv(f"{save_dir}/{task_info['task_name']}.csv")

    yraw = np.array(schedule.y.values)
    t_r = schedule[["Time", "R_before"]]
    bounds = experiment_config['tune_config']['scheduler']._hyperparam_bounds
    hparams = schedule[bounds.keys()]
    Xraw = pd.concat([t_r, hparams], axis=1).values

    # logscale hyperparameters
    if hasattr(experiment_config['tune_config']['scheduler'], '_log_scale_hyperparam'):
        log_scale_hyperparam = experiment_config['tune_config']['scheduler']._log_scale_hyperparam
    else:
        log_scale_hyperparam = []
    X, y = transform_meta_data(Xraw, yraw, bounds, log_scale_hyperparam)

    m, _ = pb2_gp(X, y, None)

    m.pickle(f"{save_dir}/{task_info['task_name']}.pickle")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs experiment defined in configuration file.')
    parser.add_argument('--moab_id', default='0', type=int,
                        help="Specifies which experiment configuration to run")
    parser.add_argument('--config_file_name', type=str,
                        help='Name of the configuration file of the experiment')
    parser.add_argument('--experiment_dir', type=str,
                        help='Name of directory containing the configs')
    parser.add_argument('--save_meta', action="store_true", default=False,
                        help="Savees data for later use in metaPB2 methods.")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir

    results_metrics, experiment_config, errors = utils.run_configuration(
        experiment_dir + '/' + args.config_file_name,
        int(args.moab_id))
    if args.save_meta:
        save_schedules(experiment_config, args.config_file_name)
        # save_gaussian_process(experiment_config)
    save_experiment(results_metrics, experiment_config, errors)
