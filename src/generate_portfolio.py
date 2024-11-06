"""
Last step of the portfolio creation algorithm.
Create a performance matrix and generate portfolios for each task based on the rest of the tasks.
"""

import argparse
import ast
import copy
import json
import os
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches


def format_missing_runs(missing_runs: list) -> None:
    """
    Formats the missing runs into a more readable variant for rerunning with slurm and prints it.
    Args:
        missing_runs (list): List of integers representing the IDs of the missing runs.

    Returns:
        None
    """
    if len(missing_runs) == 0:
        return
    missing_runs = missing_runs.copy()
    new_missing = ""
    current_id = missing_runs.pop(0)
    starting_id = current_id
    for run_id in missing_runs:
        if run_id == current_id + 1:
            current_id = run_id
        else:
            if starting_id == current_id:
                new_missing += f"{starting_id},"
            else:
                new_missing += f"{starting_id}-{current_id},"

            current_id = run_id
            starting_id = run_id

    if run_id == starting_id:
        new_missing += str(starting_id)
    else:
        new_missing += f"{starting_id}-{run_id}"
    print(new_missing)


def generate_portfolio(initial_configs_save_dir: str,
                       wanted_portfolio_size: int,
                       same_environment_filter: bool = False,
                       save: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generates a portfolio of configurations based on the performance matrix of the configurations in the initial_configs_save_dir.
    The portfolio is saved in a json file in the parent directory of initial_configs_save_dir.
    Additionally, a visualization of the portfolios is generated and returned.

    Args:
        initial_configs_save_dir (str): Path to the directory containing the initial configuration files.
        wanted_portfolio_size (int): Number of configurations to include in the portfolio.
        same_environment_filter (bool, optional): If True, only use the same base environments (different gravities) to construct the portfolio. Defaults to False.
        save (bool, optional): If True, save the generated portfolio to a file. Defaults to True.

    Returns:
        tuple: A tuple containing the figure and axis of the generated plot.
    """
    reordered = True  # Reorder the x-axis based on the string name of the environments.
    display_style = 'point'  # Marks selected configurations with a point

    json_list = get_json_files(initial_configs_save_dir)
    stats_dict = read_stats(initial_configs_save_dir)
    num_settings = stats_dict["num_settings"]
    num_envs = stats_dict["num_envs"]
    mode = stats_dict["mode"]
    env_map_list, settings_map_list, source_env_map_list = np.array([]), np.array([]), np.array(
        [])  # this will be used to map envs/settings to indices in perf matrix
    performance_matrix = np.zeros((num_settings, num_envs))
    executed_ids = [int((result_path.split('/')[-1]).split('.')[0]) for result_path in json_list]
    missing_runs = [run_id for run_id in range(num_settings * num_envs) if run_id not in executed_ids]
    print('Number of missing runs: {}'.format(len(missing_runs)))
    print('Missing run ids: ', missing_runs)
    format_missing_runs(missing_runs)
    for json_file in json_list:
        with open(json_file, 'r') as file:
            rerun_result = json.load(file)
        score = rerun_result['score']
        env = rerun_result['run_env']
        source_env = rerun_result['source_env'].removesuffix(
            '.csv')  # removesuffix should not be needed after rerun fix
        hp_setting = json.dumps(rerun_result["hyperparameters"])
        if not np.isin(env, env_map_list):
            env_map_list = np.append(env_map_list, env)
        if not np.isin(hp_setting, settings_map_list):
            settings_map_list = np.append(settings_map_list, [hp_setting])
            source_env_map_list = np.append(source_env_map_list, source_env)
        env_id = np.where(env_map_list == env)[0][0]
        setting_id = np.where(settings_map_list == hp_setting)[0][0]

        performance_matrix[setting_id, env_id] = score

    min_perf = np.min(performance_matrix, axis=0)
    max_perf = np.max(performance_matrix, axis=0)
    # normalised
    performance_matrix_norm = (performance_matrix - min_perf) / (max_perf - min_perf + 1e-8)

    # maybe check for dimensions and adjust it based on that

    # number of columns in the matrix
    n_cols = performance_matrix_norm.shape[1]
    # maybe I should just go with an if statement here
    fig, ax = plt.subplots(figsize=(n_cols * 0.15 + 3, 7))
    if reordered:
        viz_order = np.argsort(np.array(env_map_list))
        source_viz_order = np.argsort(np.array(source_env_map_list))
        viz_matrix = performance_matrix_norm[source_viz_order, :][:, viz_order]
        fig.colorbar(ax.imshow(viz_matrix, aspect="auto", cmap="Blues")).set_label('Normalized Performance')
    else:
        fig.colorbar(ax.imshow(performance_matrix_norm, aspect="auto", cmap="Blues")).set_label(
            'Normalized Performance')

    # normalise performance, because not all problems are on same scale
    # use distance to best
    if mode == "min":
        filter_func = np.min
    else:
        filter_func = np.max
    # create different portfolios
    for idx, portfolio_env in enumerate(env_map_list):
        # remove the current environment from the environments and remove all settings originating from that env
        env_mask = env_map_list != portfolio_env
        setting_mask = source_env_map_list != portfolio_env

        if same_environment_filter:
            same_env_mask = np.array([' '.join(portfolio_env.split(' ')[:-1]) in env for env in env_map_list])
            env_mask = np.logical_and(env_mask, same_env_mask)

        assert not env_mask.all()
        assert not setting_mask.all()

        current_settings_map_list = settings_map_list[setting_mask]
        current_performance_matrix = performance_matrix[setting_mask][:, env_mask]

        min_perf = np.min(current_performance_matrix, axis=0)
        max_perf = np.max(current_performance_matrix, axis=0)
        # normalised
        performance_matrix_norm = (current_performance_matrix - min_perf) / (max_perf - min_perf + 1e-8)
        # ready to create portfolio using greedy submodular
        portfolio_ids = []
        portfolio_ids_visualization = []
        portfolio = []
        if wanted_portfolio_size > len(performance_matrix_norm):
            raise ValueError('wanted portfolio size is greater than the number of configurations provided')
        settings_map_list_copy = copy.deepcopy(current_settings_map_list)
        # print(settings_map_list_copy)
        while len(portfolio_ids) < wanted_portfolio_size:
            score_map = {}
            for hp_setting in settings_map_list_copy:
                setting_id = np.where(current_settings_map_list == hp_setting)[0][0]
                # new score IF current HP is added
                score_current = np.sum(filter_func(performance_matrix_norm[[portfolio_ids + [setting_id]]], axis=0))
                score_map[hp_setting] = score_current
            # find the hp that maxmises the performance
            next_hp = max(score_map, key=score_map.get)
            # add to portfolio
            portfolio_ids.append(np.where(current_settings_map_list == next_hp)[0][0])
            portfolio_ids_visualization.append(np.where(settings_map_list == next_hp)[0][0])
            portfolio.append(next_hp)
            # remove from copy in order to not consider for next iteration
            settings_map_list_copy = settings_map_list_copy[settings_map_list_copy != next_hp]

        if reordered:
            portfolio_ordered_index = np.arange(len(source_viz_order))[
                np.isin(source_viz_order, portfolio_ids_visualization)]
            if display_style == 'point':
                ax.scatter(
                    [np.arange(len(viz_order))[viz_order == idx][0] for _ in range(len(portfolio_ids_visualization))],
                    portfolio_ordered_index, c='red', s=30, marker='x')
            elif display_style == 'rectangle':
                for i in portfolio_ordered_index:
                    ax.add_patch(
                        patches.Rectangle((np.arange(len(viz_order))[viz_order == idx][0] - 0.5, i - 0.5), 1, 1,
                                          linewidth=1, edgecolor='r', facecolor='none'))

        if save:
            save_portfolio(portfolio, initial_configs_save_dir, portfolio_env)

    # sns.heatmap(performance_matrix_norm, annot=False, ax=ax, cmap='Blues')
    if reordered:
        ax.set_xticks(np.arange(len(env_map_list)), labels=np.array(env_map_list)[viz_order])
    else:
        ax.set_xticks(np.arange(len(env_map_list)), labels=env_map_list)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
    ax.set_ylabel(f"Configurations")
    ax.set_xlabel(f"Environments")

    plt.tight_layout()
    plt.show()

    portfolio_type = 'feat' if same_environment_filter else 'general'
    if save:
        fig.savefig(
            '/'.join(initial_configs_save_dir.split('/')[:-1]) + f"/portfolio_{display_style}_{portfolio_type}.png")
        fig.savefig(
            '/'.join(initial_configs_save_dir.split('/')[:-1]) + f"/portfolio_{display_style}_{portfolio_type}.pdf")
    return fig, ax


def save_portfolio(portfolio, initial_configs_save_dir: str, portfolio_name: str = None) -> None:
    """
    Save the portfolio to a json file.

    Args:
        portfolio (list): A list of hyperparameter configurations to be saved.
        initial_configs_save_dir (str): Path to the directory containing the initial configuration files.
        portfolio_name (str, optional): Name of the portfolio to be used in the filename. Defaults to None.

    Returns:
        None
    """
    # convert portfolio array of str representation into dict to store it
    portfolio = [ast.literal_eval(s) for s in portfolio]
    portfolio_dir = '/'.join(initial_configs_save_dir.split('/')[:-1]) + "/portfolio"
    if not os.path.exists(portfolio_dir):
        os.makedirs(portfolio_dir)

    with open(portfolio_dir + f"/{str(portfolio_name) + '_' if portfolio_name is not None else ''}portfolio.json",
              'w') as f:
        f.write(json.dumps(portfolio))


def get_json_files(directory: str) -> list:
    """
    Returns a list of paths to all the json files in the given directory.
    """
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def read_stats(initial_regions_save_dir: str) -> dict:
    """
    Reads the stats.json file and returns the stats dictionary.

    Args:
        initial_regions_save_dir (str): Path to the directory containing the stats.json file.

    Returns:
        dict: A dictionary containing the statistics from the stats.json file.
        The dictionary format is: {"num_settings": value, "num_envs": value, "mode": value}
    """
    file_path = '/'.join(initial_regions_save_dir.split('/')[:-1]) + "/stats.json"
    with open(file_path, 'r') as file:
        stats = json.load(file)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate portfolio from configs in config_dir')
    parser.add_argument('--initial_regions_save_dir', required=True, type=str,
                        help='Directory where the rerun results are saved.')
    parser.add_argument('--portfolio_size', type=int, default=2,
                        help='Number of configurations in a portfolio.')
    parser.add_argument('--same_env_filter', action='store_true',
                        help='Enable to only use the same base environments (different gravities) to construct '
                             'the portfolio')
    args = parser.parse_args()
    portfolio = generate_portfolio(str(args.initial_regions_save_dir), int(args.portfolio_size), args.same_env_filter)
