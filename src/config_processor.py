import copy
import datetime
import itertools
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
from ray.tune.search import BasicVariantGenerator

from src.algorithms.TAF_pb2 import TAF_PB2
from src.algorithms.log_pb2 import LogPB2
from src.algorithms.meta_prior_pb2 import MetaPriorPB2
from src.algorithms.multi_task_pb2 import MultiTaskPB2
from src.rl.env_classes import register_carl_env


# when the config passed is an object instead of a filepath
# useful in case of running inirial configurations to avoid duplication of files
def process_config_object(config_object, experiment_idx, config_file_name):
    experiment_config = configure_experiment(config_object, experiment_idx, config_file_name)
    return experiment_config


def process_configs(config_file, experiment_idx):
    # loading configuration from python file by importing it as a module
    config_file_path = os.path.dirname(str(Path(__file__).absolute().parent.parent) + "/" + config_file)
    sys.path.insert(1, config_file_path)
    import importlib
    config = importlib.import_module(config_file.split("/")[-1], package=None)

    experiment_config = configure_experiment(config, experiment_idx, config_file)
    return experiment_config


def configure_experiment(config, experiment_idx, config_file):
    experiment_config = copy.deepcopy(
        create_experiment_config(config.combined_config, experiment_idx))  # copy to not mess stuff up

    experiment_config['seed'] = experiment_config['seed']['seed']
    # add file name and trial name to the config
    experiment_config['trial_name'] = config.get_trial_name(experiment_config)

    # TUNE Config Stuff
    tune_config = experiment_config['tune_config']

    if "scheduler" in tune_config:
        if 'use_hp_initializer' in tune_config['scheduler']:
            use_hp_initializer = tune_config['scheduler'].pop('use_hp_initializer')
            if use_hp_initializer:
                tune_config['scheduler']['hp_initializer'] = get_hp_initializer(experiment_config)

        if "perturbations" in experiment_config:  # if we want to test the same exp but across diff perturbation intervals
            tune_config["scheduler"].update(
                {"perturbation_interval": experiment_config["perturbation_interval"]["perturbations"]})
        if tune_config['scheduler']['type'].lower() == 'logpb2':
            # remove type from dict to pass correct arguments for Scheduler
            tune_config['scheduler'].pop("type")
            tune_config['scheduler'] = LogPB2(**tune_config['scheduler'])
        elif tune_config['scheduler']['type'].lower() == 'metapriorpb2':
            tune_config['scheduler'].pop("type")
            tune_config['scheduler']['meta_directory'] = tune_config['scheduler'][
                                                             'meta_directory'] + f"/{experiment_config['seed']}"
            tune_config['scheduler']['excluded_metadata'] = str(
                json.loads(experiment_config['trial_name'])['task_name'])
            tune_config['scheduler'] = MetaPriorPB2(**tune_config['scheduler'])
        elif tune_config['scheduler']['type'].lower() == 'multitaskpb2':
            tune_config['scheduler'].pop("type")
            tune_config['scheduler']['meta_directory'] = tune_config['scheduler'][
                                                             'meta_directory'] + f"/{experiment_config['seed']}"
            tune_config['scheduler']['excluded_metadata'] = str(
                json.loads(experiment_config['trial_name'])['task_name'])
            tune_config['scheduler'] = MultiTaskPB2(**tune_config['scheduler'])
        elif tune_config['scheduler']['type'].lower() == 'tafpb2':
            tune_config['scheduler'].pop("type")
            tune_config['scheduler']['meta_directory'] = tune_config['scheduler'][
                                                             'meta_directory'] + f"/{experiment_config['seed']}"
            tune_config['scheduler']['excluded_metadata'] = str(
                json.loads(experiment_config['trial_name'])['task_name'])
            tune_config['scheduler'] = TAF_PB2(**tune_config['scheduler'])
        elif tune_config['scheduler']['type'].lower() == 'pb2':
            tune_config['scheduler'].pop("type")
            tune_config['scheduler'] = PB2(**tune_config['scheduler'])
        elif tune_config['scheduler']['type'].lower() == 'pbt':
            tune_config['scheduler'].pop("type")
            tune_config['scheduler'] = PopulationBasedTraining(**tune_config['scheduler'])
        elif tune_config['scheduler']['type'].lower() == 'rs':
            tune_config.pop('scheduler')
            tune_config['search_alg'] = BasicVariantGenerator(random_state=experiment_config['seed'])
        else:
            raise ValueError(f"Scheduler type {tune_config['scheduler']['type']} is unknown.")

    # RUN Config Stuff
    run_config = experiment_config['run_config']
    # timelog = str(datetime.datetime.now().strftime("%Y-%m-%d"))
    run_config["local_dir"] = '/'.join(config_file.split("/")[-2:]) # + "_" + timelog
    run_config["name"] = str(experiment_idx) + '_' + config_file.split('/')[-1]

    experiment_config['algorithm'] = experiment_config['algorithm']['algorithm']
    # env configs

    experiment_config["hyperparameters"] = config.hyperparameters  # Change later

    # ENV Config Stuff

    if "env" in experiment_config["param_space"]:
        env_name = experiment_config["param_space"]["env"]
        # register does nothing if the environment is not a carl env (it should already be registered in that case)
        register_carl_env(env_name)

    return experiment_config


def create_experiment_config(config_blueprint, experiment_idx):
    # unroll

    unrolled_values = []
    assert isinstance(config_blueprint, OrderedDict), f"Val: {config_blueprint} is not a OrderedDict."
    for config_type, config_dict in config_blueprint.items():
        assert isinstance(config_dict, OrderedDict)
        for key, val in config_dict.items():
            assert isinstance(val, list), f"Val: {val} is not a list."
            unrolled_values.append(val)
    # cartesian
    unrolled_cartesian = list(itertools.product(*unrolled_values))

    config_to_run = unrolled_cartesian[experiment_idx]

    # construct config for experiment using blueprint as template
    experiment_configs = {}
    for key in config_blueprint:
        experiment_configs[key] = {}

    idx = 0
    for config_type, config_dict in config_blueprint.items():
        for key in config_dict:
            val = config_to_run[idx]
            experiment_configs[config_type][key] = val
            idx += 1

    return experiment_configs


def get_hp_initializer(experiment_config):
    hyperparameters = list(experiment_config['tune_config']['scheduler']['hyperparam_bounds'].keys())
    param_space = experiment_config['param_space']

    def hp_initializer():
        return [param_space[hp].sample() for hp in hyperparameters]

    return hp_initializer
