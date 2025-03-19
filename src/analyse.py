# load the stuff
# merge it somehow
# create pictures
import argparse
import glob
import json
import logging
import warnings
from itertools import combinations
from math import ceil
from pathlib import Path
from typing import List, Union
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from autorank import autorank, create_report, plot_stats
from matplotlib import colors
from matplotlib.ticker import MultipleLocator
from scipy import stats
from scipy.stats import rankdata

from src.analyse_utils import plot_curves_with_ranked_legends_paper, DEFAULT_MARKER_KWARGS, MARKERS, plot_mean_std, \
    COLORS
from src.utils import get_src_path


def get_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def load_multiple_results(results_paths: List[str], dl_split: str = 'test'):
    assert dl_split in ['test', 'valid', 'rl-eval', 'rl-train', 'train_loss', 'train_score']
    results_dfs = []
    for dir in results_paths:
        results_path = get_src_path().parent / 'ray_results' / dir
        experiment_dataframe = load_results(results_path, dl_split)
        results_dfs.append(experiment_dataframe)
    experiment_dataframe = pd.concat(results_dfs, axis=0).reset_index(drop=True)

    return experiment_dataframe


def rl_metric_rename(results_metrics, old_metric, new_metric, nan_col=True):
    reward_columns = [column for column in results_metrics.columns if old_metric in column]
    metric_duplicates = results_metrics[reward_columns]
    # Todo: they both are identical
    idx_missing = np.argmax(metric_duplicates.isnull().any().values)
    results_metrics[new_metric] = metric_duplicates.iloc[:, idx_missing]
    return results_metrics


def load_results(results_path: Union[str, Path], dl_split: str):
    dir_path = str(results_path) + r'/*_result.csv'
    dataframes = []
    for file in glob.glob(dir_path):
        dataframes.append(pd.read_csv(file))
        has_nan = pd.read_csv(file).isna().any().any()
        if has_nan:
            Warning("file {file} has nans".format(file=file))
    try:
        experiment_dataframe = pd.concat(dataframes, axis=0).reset_index(drop=True)
    except ValueError as e:
        print('Value error occurred for the following result dir: ', results_path)
        raise e
    experiment_dataframe.drop(columns='score', errors='ignore', inplace=True)
    if '_training_iteration' in experiment_dataframe.columns:
        experiment_dataframe['_training_iteration'] = experiment_dataframe['_training_iteration'] - 1
    if dl_split == 'rl-eval':
        # i have to get the right one here
        old_metric = 'episode_reward_mean'
        experiment_dataframe = rl_metric_rename(experiment_dataframe, old_metric, 'valid_score', nan_col=True)
        experiment_dataframe = rl_metric_rename(experiment_dataframe, old_metric, 'score', nan_col=True)
        # drop all the nan values
        experiment_dataframe = experiment_dataframe[experiment_dataframe['score'].notna()]

        # get the smallest training iteration
        smallest_timestep = experiment_dataframe['timesteps_total'].min()
        experiment_dataframe.drop(columns='iteration')
        experiment_dataframe['iteration'] = experiment_dataframe['timesteps_total'].map(lambda x: x / smallest_timestep)
        experiment_dataframe['iteration'] = experiment_dataframe['iteration'] - 1
        # filter for only integer thingies
        experiment_dataframe = experiment_dataframe[experiment_dataframe['iteration'] % 1 == 0]
        experiment_dataframe['iteration'] = experiment_dataframe['iteration'].astype(int)

    if dl_split == 'rl-train':
        raise NotImplementedError()
        old_metric = 'episode_reward_mean'
        experiment_dataframe = rl_metric_rename(experiment_dataframe, old_metric, 'valid_score', nan_col=False)
        experiment_dataframe = rl_metric_rename(experiment_dataframe, old_metric, 'score', nan_col=False)

    elif dl_split == 'train_loss':
        if 'loss_train' not in experiment_dataframe:
            return pd.DataFrame()
        experiment_dataframe['valid_score'] = experiment_dataframe['loss_train']
        experiment_dataframe['score'] = experiment_dataframe['loss_train']
        experiment_dataframe = experiment_dataframe[experiment_dataframe['score'].notna()]
    elif dl_split == 'train_score':
        if 'roc_auc_train' not in experiment_dataframe:
            return pd.DataFrame()
        experiment_dataframe['valid_score'] = experiment_dataframe['roc_auc_train']
        experiment_dataframe['score'] = experiment_dataframe['roc_auc_train']
        experiment_dataframe = experiment_dataframe[experiment_dataframe['score'].notna()]
    elif dl_split == 'valid':
        experiment_dataframe['valid_score'] = experiment_dataframe['roc_auc_valid']
        experiment_dataframe['score'] = experiment_dataframe['roc_auc_valid']
    elif dl_split == 'test':
        experiment_dataframe['valid_score'] = experiment_dataframe['roc_auc_valid']
        experiment_dataframe['score'] = experiment_dataframe['roc_auc_test']

    if dl_split in ['test', 'valid', 'train_loss', 'train_score']:
        experiment_dataframe.rename(columns={'_training_iteration': 'iteration'}, errors='ignore', inplace=True)
        experiment_dataframe = experiment_dataframe.loc[experiment_dataframe['iteration'] < 15]
    return experiment_dataframe

def get_hue(dataframe):
    normalized_score = np.zeros(len(dataframe))
    for iteration in pd.unique(dataframe['iteration']):
        mask = dataframe['iteration'] == iteration
        values = dataframe.loc[mask, 'score']
        normalized_score[mask] = (values - np.min(values)) / (np.max(values) - np.min(values))
    return normalized_score

# training plot
def training_plot(dataset_group: pd.DataFrame, dataset_name: str):
    """
    Plots mean learning curves and their uncertainty over seeds for each method.
    :param results_path: path to result files of experiment
    """
    fig, ax = plt.subplots()

    grouped = dataset_group.groupby(["algorithm"])

    for name, group in grouped:
        name = name[0]

        def performance_max_worker(x):
            max_ind = np.argmax(x['valid_score'])
            score = x['score'].iloc[max_ind]
            return pd.Series({'score': score})

        max_reward_per_iteration_seed = group.groupby(["iteration", "seed"]).apply(
            performance_max_worker).reset_index()
        reward_statistics = max_reward_per_iteration_seed.groupby(['iteration']).score.agg(
            ["mean", stats.sem]).reset_index()
        ax.plot(reward_statistics['iteration'], reward_statistics['mean'], label=name)  # , drawstyle='steps-mid'
        ax.fill_between(reward_statistics['iteration'], reward_statistics['mean'] - reward_statistics['sem'],
                        reward_statistics['mean'] + reward_statistics['sem'], alpha=0.5)  # , step='mid'
    ax.set_xlabel("Time Steps")
    ax.set_ylabel('Reward')
    ax.legend()
    ax.set_title(f'Score over Time on Dataset {dataset_name}')
    return fig, ax


def training_plot_per_dataset(experiment_dataframe: pd.DataFrame, output_path: Path):
    """
    Plots mean learning curves and their uncertainty over seeds for each method.
    :param results_path: path to result files of experiment
    """
    experiment_dataframe['dataset'] = experiment_dataframe["trial_name"].map(lambda x: json.loads(x)['task_name'])
    experiment_dataframe['algorithm'] = experiment_dataframe["trial_name"].map(lambda x: json.loads(x)['algorithm'])
    # experiment_dataframe = experiment_dataframe.loc[experiment_dataframe['iteration'] < 15]
    dataset_grouped = experiment_dataframe.groupby(['dataset'])

    for dataset_name, dataset_group in dataset_grouped:
        fig, ax = training_plot(dataset_group, dataset_name[0])
        save_path = Path(output_path) / 'training_plots' / f'{dataset_name[0]}.png'
        save_path_pdf = Path(output_path) / 'training_plots' / f'{dataset_name[0]}.pdf'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path))
        fig.savefig(str(save_path_pdf))
        plt.close(fig)

def comparison_table(experiment_dataframe: pd.DataFrame, output_path: Path, n_timesteps: int = 20):
    """
    Creates a table with the mean performance of each method for all the datasets.
    Saves the table as a latex table and as a csv.
    :param results_path:
    :param n_timesteps:
    """
    round = 2
    save_path = output_path / 'score_tables'
    save_path.mkdir(parents=True, exist_ok=True)

    max_iteration = n_timesteps - 1
    experiment_dataframe = experiment_dataframe[experiment_dataframe['iteration'] == max_iteration].copy()
    experiment_dataframe['dataset'] = experiment_dataframe["trial_name"].map(lambda x: json.loads(x)['task_name'])
    experiment_dataframe['algorithm'] = experiment_dataframe["trial_name"].map(lambda x: json.loads(x)['algorithm'])
    experiment_dataframe['dataset'] = experiment_dataframe["dataset"].map(lambda x: x.replace('_', chr(92) + '_'))

    scores = experiment_dataframe.groupby(["trial_name", "seed"]).apply(cd_performance_max_worker).reset_index()
    mean_scores = scores.groupby(["trial_name"]).agg(
        {'score': 'mean', 'dataset': lambda x: x.iloc[0], 'algorithm': lambda x: x.iloc[0]})
    std_scores = scores.groupby(["trial_name"]).agg(
        {'score': stats.sem, 'dataset': lambda x: x.iloc[0], 'algorithm': lambda x: x.iloc[0]})

    table = mean_scores.pivot(index='dataset', columns='algorithm', values='score').round(round)
    std_table = std_scores.pivot(index='dataset', columns='algorithm', values='score').round(round)
    std_table.loc['mean'] = std_table.mean()
    table.loc['mean'] = table.mean()
    # max entries pro row
    upper_bound = table + std_table
    lower_bound = table - std_table

    table_str = table.applymap(lambda x: f'{x:.{round}f}') + std_table.applymap(lambda x: f'\u00B1{x:.{round}f}')

    style = table_str.style
    for column in table.columns:
        underline_values = table[
            lower_bound[column] > upper_bound.loc[:, upper_bound.columns != column].max(axis=1)].index
        bold_values = table[table[column] > table.loc[:, table.columns != column].max(axis=1)].index

        for underline_value in underline_values:
            style = style.set_properties(subset=(underline_value, column), **{'underline': '--latex--rwrap'})
        for bold_value in bold_values:
            style = style.set_properties(subset=(bold_value, column), **{'font-weight': 'bold'})

    table_str.to_csv(save_path / f'score_table_{n_timesteps}.csv')
    style.to_latex(save_path / f'score_table_{n_timesteps}.txt', convert_css=True, hrules=True)
    print(table_str)


def critical_difference(full_experiment_dataframe: pd.DataFrame, output_path: Path, max_n_time_steps: int = 15):
    save_path = output_path / 'critical_differences_old'
    save_path.mkdir(parents=True, exist_ok=True)

    full_experiment_dataframe['dataset'] = full_experiment_dataframe["trial_name"].map(
        lambda x: json.loads(x)['task_name'])
    full_experiment_dataframe['algorithm'] = full_experiment_dataframe["trial_name"].map(
        lambda x: json.loads(x)['algorithm'])

    rank_over_time = []
    cds_over_time = []
    combined_tables = []
    indices = [i for i in range(3, max_n_time_steps)]
    for max_iteration in indices:
        n_time_steps = max_iteration + 1
        experiment_dataframe = full_experiment_dataframe[full_experiment_dataframe['iteration'] == max_iteration].copy()

        scores = experiment_dataframe.groupby(["trial_name", "seed"]).apply(cd_performance_max_worker).reset_index()
        # scores = scores[scores['seed'] < max_seeds] # not needed because filtered with dropna()
        scores['custom_index'] = scores['dataset'].map(str) + scores['seed'].map(str)  # maybe with apply
        table = scores.pivot(index='custom_index', columns='algorithm', values='score').dropna()
        combined_tables.append(table)
        result = autorank(table, alpha=0.05)
        rank_over_time.append(result.rankdf['meanrank'].rename(n_time_steps))
        cds_over_time.append(result.cd)
        print(result)
        create_report(result)
        fig, ax = plt.subplots()
        plot_stats(result, ax=ax, allow_insignificant=True)
        ax.set_title(f"Critical Difference after {n_time_steps} iterations")

        save_path = output_path / 'critical_differences' / str(n_time_steps)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path))
        plt.close(fig)

        # order by columns
        # np.repeat(np.expand_dims(result.rankdf['meanrank'], axis=1), len(result.rankdf['meanrank']), axis=1)
        # np.repeat(np.expand_dims(result.rankdf['meanrank'], axis=1), len(result.rankdf['meanrank']), axis=1).T

    combined_tables = pd.concat(combined_tables, axis=0).reset_index(drop=True)
    result = autorank(combined_tables, alpha=0.05)
    create_report(result)
    fig, ax = plt.subplots()
    plot_stats(result, ax=ax)
    ax.set_title(f"Combined Critical Difference")

    save_path = output_path / 'critical_differences' / 'combined'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    # Then function

    fig, ax = plt.subplots()
    visualize_rank_over_time(pd.DataFrame(rank_over_time), np.array(cds_over_time), ax)
    # ax.set_title('Random Search vs PB2 - Validation Split')
    fig.tight_layout()

    save_path = output_path / 'critical_differences' / 'rank_over_time'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.show()
    plt.close(fig)


def visualize_rank_over_time(rank_over_time, cds_over_time, ax):
    # iterate for algorithms
    for algorithm in rank_over_time.columns:
        ax.plot(rank_over_time.index, rank_over_time[algorithm], label=algorithm)
        ax.fill_between(rank_over_time.index, rank_over_time[algorithm] - cds_over_time / 2,
                        rank_over_time[algorithm] + cds_over_time / 2, alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Rank')
    ax.legend()


def cd_performance_max_worker(x):
    max_ind = np.argmax(x['valid_score'])
    score = x['score'].iloc[max_ind]
    d = {
        'algorithm': x['algorithm'].iloc[0],
        'dataset': x['dataset'].iloc[0],
        'score': score
    }
    return pd.Series(d)


def normalized_score(full_experiment_dataframe: pd.DataFrame, output_path: Union[Path, None] = None,
                     max_n_time_steps: int = 15, information=False):
    full_experiment_dataframe['dataset'] = full_experiment_dataframe["trial_name"].map(
        lambda x: json.loads(x)['task_name'])
    full_experiment_dataframe['algorithm'] = full_experiment_dataframe["trial_name"].map(
        lambda x: json.loads(x)['algorithm'])

    combined_tables = []
    mean_score = []
    std_score = []
    sem_score = []

    for max_iteration in range(max_n_time_steps):
        n_time_steps = max_iteration + 1
        experiment_dataframe = full_experiment_dataframe[full_experiment_dataframe['iteration'] == max_iteration].copy()

        scores = experiment_dataframe.groupby(["trial_name", "seed"]).apply(cd_performance_max_worker).reset_index()
        scores = scores.groupby(['trial_name']).agg(
            {'score': 'mean', 'dataset': lambda x: x.iloc[0], 'algorithm': lambda x: x.iloc[0]}).reset_index()
        # scores = scores[scores['seed'] < max_seeds]
        scores['custom_index'] = scores['dataset'].map(str) + '_' + str(n_time_steps)
        table = scores.pivot(index='custom_index', columns='algorithm', values='score').dropna()

        def normalize_row(row):
            min_val = row.min()
            max_val = row.max()
            return (row - min_val) / (max_val - min_val + 1e-8)

        table = table.apply(normalize_row, axis=1)
        combined_tables.append(table)
        mean_score.append(table.mean())
        std_score.append(table.std())
        sem_score.append(table.sem())
        # mean
        # std or error?

    mean_score = pd.concat(mean_score, axis=1)
    std_score = pd.concat(std_score, axis=1)
    sem_score = pd.concat(sem_score, axis=1)
    if information:
        df = mean_score.reset_index(drop=False)
        df['score'] = df[mean_score.columns.max()]
        fig, ax = plt.subplots()
        visualize_norm_matrix_paper(df, ax)
        plt.show()
        if output_path is not None:
            save_path = output_path / 'normalized_scores' / 'norm_matrix.pdf'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path))
            plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(9, 7))
        x = np.arange(1, max_n_time_steps + 1)

        for count, index in enumerate(mean_score.index):
            linewidth = 3
            color = COLORS[count % len(COLORS)]
            marker = MARKERS[count % len(MARKERS)]
            marker_kwargs = DEFAULT_MARKER_KWARGS
            markers_on = np.arange(len(x))
            y = np.array([np.array(table[index]) for table in combined_tables]).T
            n_std = 1 / np.sqrt(len(y))
            alpha = 0.3
            plot_mean_std(
                x, y, linewidth=linewidth, ax=ax, color=color, alpha=alpha, n_std=n_std,
                marker=marker, markevery=markers_on, **marker_kwargs, label=index
            )

        # for index in mean_score.index:
        #     y = mean_score.loc[index]
        #     uncertainty_lb, uncertainty_ub = y-sem_score.loc[index], y+sem_score.loc[index]
        #     ax.plot(x, y, label=index)
        #     ax.fill_between(x, uncertainty_lb, uncertainty_ub , alpha=.3)

        ax.set_ylabel(f"Normalized score", fontsize=22)
        ax.set_xlabel("Number of iterations", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        xlim_min, xlim_max = ax.get_xlim()
        ax.set_xlim([0, xlim_max])
        # ax.legend(loc='upper center', ncol=3, fontsize=14,
        #           bbox_to_anchor=(0.5, 1.05))  # (bbox_to_anchor=(1.04, 1), loc="upper left")

        plt.show()
        if output_path is not None:
            save_path = output_path / 'normalized_scores' / 'paper.pdf'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path))
            fig.savefig(str(output_path / 'normalized_scores' / 'paper.png'))

            plt.close(fig)


def visualize_norm_matrix_paper(df, ax):
    df['Algorithm'] = df['algorithm'].map(lambda x: x.split('+')[0])
    df['Combined seeds'] = df['algorithm'].map(lambda x: int(x.split('+')[1]))
    tc = df.pivot(index='Combined seeds', columns='Algorithm', values='score').sort_index(ascending=False)
    sns.heatmap(tc, annot=True, ax=ax, cmap='Blues')


def critical_difference_paper(full_experiment_dataframe: pd.DataFrame, output_path: Union[Path, None] = None,
                              max_n_time_steps: int = 15, information=False):
    full_experiment_dataframe['dataset'] = full_experiment_dataframe["trial_name"].map(
        lambda x: json.loads(x)['task_name'])
    full_experiment_dataframe['algorithm'] = full_experiment_dataframe["trial_name"].map(
        lambda x: json.loads(x)['algorithm'])

    combined_ranks = []
    combined_tables = []
    for max_iteration in range(max_n_time_steps):
        n_time_steps = max_iteration + 1
        experiment_dataframe = full_experiment_dataframe[full_experiment_dataframe['iteration'] == max_iteration].copy()

        scores = experiment_dataframe.groupby(["trial_name", "seed"]).apply(cd_performance_max_worker).reset_index()
        scores = scores.groupby(['trial_name']).agg(
            {'score': 'mean', 'dataset': lambda x: x.iloc[0], 'algorithm': lambda x: x.iloc[0]}).reset_index()
        # scores = scores[scores['seed'] < max_seeds]
        scores['custom_index'] = scores['dataset'].map(str) + '_' + str(n_time_steps)
        table = scores.pivot(index='custom_index', columns='algorithm', values='score').dropna()
        combined_tables.append(table)
        combined_ranks.append(pd.DataFrame(rankdata(-table, axis=1), columns=table.columns))

    stat_sig_last_iteration = autorank(combined_tables[-1], alpha=0.05)
    sig_map_last_iteration = get_significance_map(stat_sig_last_iteration)

    combined_tables = pd.concat(combined_tables, axis=0).reset_index(drop=True)
    result = autorank(combined_tables, alpha=0.05)
    sig_map = get_significance_map(result)
    cd = result.cd

    data_x = np.arange(1, len(combined_ranks) + 1)
    data_y = {}
    for algorithm in combined_tables.columns:
        values = []
        for df in combined_ranks:
            values.append(df[algorithm])
        data_y[algorithm] = pd.concat(values, axis=1).values

    if information:
        fig, ax = plt.subplots()
        df = stat_sig_last_iteration.rankdf.reset_index(drop=False)
        visualize_rank_matrix_paper(df, ax)
        plt.show()

        if output_path is not None:
            save_path = output_path / 'critical_differences' / 'rank_matrix.pdf'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path))
            plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(14, 7))
        plot_curves_with_ranked_legends_paper(
            ax=ax,
            data_y=data_y,
            data_x=data_x,
            show_std_error=True,
            min_is_the_best=True,
            stat_significance_map=sig_map,
            cd=cd
        )
        ax.set_ylabel(f"Rank (out of {len(combined_tables.columns)})", fontsize=24)
        ax.set_xlabel("Number of iterations", fontsize=22)
        ax.xaxis.set_label_coords(.3, -0.06)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlim([0, max_n_time_steps + 10])
        plt.show()

        if output_path is not None:
            save_path = output_path / 'critical_differences' / 'paper.pdf'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path))
            fig.savefig(str(output_path / 'critical_differences' / 'paper.png'))
            plt.close(fig)
        # mapping each method to an matrix (trials, iterations)


def visualize_rank_matrix_paper(df, ax):
    df['Algorithm'] = df['algorithm'].map(lambda x: x.split('+')[0])
    df['Combined seeds'] = df['algorithm'].map(lambda x: int(x.split('+')[1]))
    tc = df.pivot(index='Combined seeds', columns='Algorithm', values='meanrank').sort_index(ascending=False)
    sns.heatmap(tc, annot=True, ax=ax, cmap='Blues')


def get_significance_map(results_autorank):
    cd = results_autorank.cd
    if cd is None:
        logging.warning("The critical difference was None")

        return False
    ranks = results_autorank.rankdf['meanrank']
    return lambda alg_1, alg_2: abs(ranks[alg_1] - ranks[alg_2]) > cd

def visualize_all_workers(experiment_dataframe: pd.DataFrame, results_path: Path, output_path: Path):
    for algorithm in pd.unique(experiment_dataframe["trial_name"].map(lambda x: json.loads(x)['algorithm'])):
        for dataset in pd.unique(experiment_dataframe["trial_name"].map(lambda x: str(json.loads(x)['task_name']))):
            visualize_workers(experiment_dataframe.copy(), results_path, output_path, dataset, algorithm)


def visualize_workers(experiment_dataframe: pd.DataFrame, results_path: Path, output_path: Path, dataset: str,
                      algorithm: str):
    experiment_dataframe['dataset'] = experiment_dataframe["trial_name"].map(lambda x: str(json.loads(x)['task_name']))
    experiment_dataframe['algorithm'] = experiment_dataframe["trial_name"].map(lambda x: json.loads(x)['algorithm'])
    experiment_dataframe = experiment_dataframe[experiment_dataframe['algorithm'] == algorithm]
    experiment_dataframe = experiment_dataframe[experiment_dataframe['dataset'] == dataset]
    experiment_dataframe = experiment_dataframe.copy()

    with open(results_path / "info.json") as f:
        info_file = json.load(f)
        hyperparameters = info_file["hyperparameters"]
    seeds = sorted(list(pd.unique(experiment_dataframe['seed'])))

    fig, ax = plt.subplots(len(hyperparameters) + 1, len(seeds), sharex='all', sharey='row', squeeze=False,
                           figsize=(len(seeds) * 4., 4.5 * (len(hyperparameters) + 1)))
    # test the integer dataset value

    # plot the learning curve
    # plot the hyperparameter
    for idx_seed, seed in enumerate(seeds):
        hyperparameter_schedule = experiment_dataframe[experiment_dataframe['seed'] == seed]
        for trial_id in pd.unique(hyperparameter_schedule['trial_id']):
            worker_schedule = hyperparameter_schedule[hyperparameter_schedule['trial_id'] == trial_id]
            worker_schedule = worker_schedule[worker_schedule['iteration'] > 0]
            ax[0, idx_seed].plot(worker_schedule['iteration'],
                                 worker_schedule['score'],
                                 drawstyle='steps-post', label=trial_id)
        ax[0, idx_seed].set_xlabel('Time steps')
        ax[0, idx_seed].set_ylabel('Validation Score')

        if idx_seed == 0:
            in_data = experiment_dataframe['score']
            near_min = in_data.quantile(0.025)
            near_max = in_data.quantile(0.975)
            range = near_max - near_min
            ax[0, 0].set_ylim(near_min - range * 0.1, near_max + range * 0.1)

        # plot hyperparamter schedules
        for idx_hyperparameter, hyperparameter in enumerate(hyperparameters):
            for trial_id in pd.unique(hyperparameter_schedule['trial_id']):
                worker_schedule = hyperparameter_schedule[hyperparameter_schedule['trial_id'] == trial_id]
                ax[idx_hyperparameter + 1, idx_seed].plot(worker_schedule['iteration'],
                                                          worker_schedule[hyperparameter],
                                                          drawstyle='steps-post', label=trial_id)
            # check if we assume logarithm by adding it to the information file
            ax[idx_hyperparameter + 1, idx_seed].set_xlabel('Time steps')
            ax[idx_hyperparameter + 1, idx_seed].set_ylabel(hyperparameter)

            if hyperparameter in ['lr', 'learning_rate', 'weight_decay']:
                ax[idx_hyperparameter + 1, idx_seed].set_yscale('log')

    fig.suptitle(f'Algorithm: {algorithm}, Dataset: {dataset}', fontsize=32)
    save_path = output_path / 'worker_visualization' / f"{dataset}_{algorithm}.png".replace(' ', '_')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)


def load_weights(weight_dirs: list[str]):
    weight_dirs = [get_src_path().parent / 'ray_results' / weight_dir for weight_dir in weight_dirs]
    weight_dfs = []
    for weight_dir in weight_dirs:
        weight_files_regex = str(weight_dir) + r'/*_weights.csv'
        for file in glob.glob(weight_files_regex):
            weight_dfs.append(pd.read_csv(file))
    if len(weight_dfs) == 0:
        return None
    return pd.concat(weight_dfs, axis=0).reset_index(drop=True)

def visualize_weights_matrix(weight_dfs: pd.DataFrame, output_path: Union[Path,None]=None, log=True):
    weight_dfs['algorithm'] = weight_dfs["trial_name"].map(lambda x: json.loads(x)['algorithm'])
    weight_dfs['task_name'] = weight_dfs["trial_name"].map(lambda x: json.loads(x)['task_name'])
    total_weights = weight_dfs.drop(columns=['trial_name', 'seed', 'time_step'])

    target_column = total_weights.pop('target')
    for column in total_weights.columns:
        total_weights[column] = total_weights[column].fillna(target_column)

    for algorithm in pd.unique(total_weights['algorithm']):
        algorithm_df = total_weights[total_weights['algorithm'] == algorithm].drop(columns=['algorithm'])
        algorithm_df = algorithm_df.groupby(by=['task_name']).agg(np.mean)

        algorithm_df = algorithm_df.sort_index(axis=1).sort_index(axis=0)
        fig, ax = plt.subplots(figsize=(10., 10.))
        # I have to sort them.... then maybe rearrange them to show clusters
        weight_matrix(algorithm_df, ax, log)
        # fig.suptitle(f"Weight Average {algorithm}")
        fig.tight_layout()
        plt.show()
        if output_path is not None:
            save_path = output_path / "weight_matrices" / f"{algorithm}.pdf"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            plt.close(fig)


def weight_matrix(weight_df, ax, log):
    if log:
        norm_kwargs = {'vmin': 0.01, 'vmax': 1.}
        im = ax.imshow(weight_df, norm=colors.LogNorm(clip=True, **norm_kwargs), cmap='Blues')
    else:
        # norm_kwargs = {'vmin': 0., 'vmax': 1.}
        # im = ax.imshow(weight_df, norm=colors.Normalize(**norm_kwargs), cmap='Blues')
        im = ax.imshow(weight_df, cmap='Blues')
    ax.set_ylabel('Target task')
    ax.set_yticks(np.arange(len(weight_df.columns)), labels=weight_df.columns)
    ax.set_xlabel('Meta-task')
    ax.set_xticks(np.arange(len(weight_df.index)), labels=weight_df.index)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.79)  # , fraction=0.04, pad=0.038, aspect=10 # fraction=0.02,
    cbar.ax.set_ylabel('Weighting')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')


def missing_experiments(experiment_dataframe: pd.DataFrame, output_path: Path) -> None:
    save_path = output_path / 'missing_experiments'
    save_path.mkdir(parents=True, exist_ok=True)

    experiment_dataframe['dataset'] = experiment_dataframe["trial_name"].map(
        lambda x: json.loads(x)['task_name'].replace('_', chr(92) + '_'))
    experiment_dataframe['algorithm'] = experiment_dataframe["trial_name"].map(
        lambda x: json.loads(x)['algorithm'].replace('_', chr(92) + '_'))

    unique_combinations = experiment_dataframe[['dataset', 'algorithm', 'seed']].drop_duplicates()
    count = unique_combinations.groupby(by=['algorithm', 'dataset']).agg('count').reset_index()
    count['seed'] = count['seed'].max() - count['seed']
    count.rename(columns={'dataset': 'Environment', 'algorithm': 'Algorithm'}, inplace=True)
    missing_value_table = count.pivot(index='Environment', columns='Algorithm', values='seed')

    missing_value_table.loc['mean'] = missing_value_table.sum()
    missing_value_table.to_latex(save_path / 'missing_experiments.txt')


def visualize_runtime(experiment_dataframe: pd.DataFrame, output_path: Union[Path, None] = None):
    time_var = 'total_run_time'
    # run time for entire experiment
    run_time = experiment_dataframe.groupby(by=["trial_name", "seed"]).agg({time_var: 'max'}).reset_index()
    # average run tune per trial
    run_time = run_time.groupby(by=["trial_name"]).agg({time_var: 'mean'}).reset_index()
    run_time['dataset'] = run_time["trial_name"].map(lambda x: json.loads(x)['task_name'])
    run_time['algorithm'] = run_time["trial_name"].map(lambda x: json.loads(x)['algorithm'])

    run_time = run_time.dropna()

    with sns.axes_style("whitegrid"):
        run_time.rename(columns={time_var: 'Total runtime in s', 'algorithm': 'Algorithm'}, inplace=True)
        ax = sns.boxplot(x=run_time["Algorithm"], y=run_time['Total runtime in s'], linewidth=5)
        pb2_median = add_median_labels(ax)

        # Adding the secondary y-axis with different labels

        def inverse_relabel_yaxis(y):
            return y / 100 * pb2_median + pb2_median

        def relabel_yaxis(y):
            return (y - pb2_median) / pb2_median * 100

        secax = ax.secondary_yaxis('right', functions=(relabel_yaxis, inverse_relabel_yaxis))
        secax.set_ylabel('Percentage difference from PB2 median')

        # rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        if output_path is not None:
            save_dir = output_path / 'runtime'
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / "runtime.png")
            plt.savefig(save_dir / "runtime.pdf")


def add_median_labels(ax: plt.Axes, fmt: str = ".0f") -> None:
    """Add text labels to the median lines of a seaborn boxplot.

    Args:
        ax: plt.Axes, e.g. the return value of sns.boxplot()
        fmt: format string for the median value
    """
    pb2_idx = next(idx for idx, label in enumerate(ax.get_xticklabels()) if label.get_text() == 'PB2')

    import matplotlib.patheffects as path_effects
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if "Patch" in str(c)]
    start = 4
    if not boxes:  # seaborn v0.13 => fill=False => no patches => +1 line
        boxes = [c for c in ax.get_lines() if len(c.get_xdata()) == 5]
        start += 1
    lines_per_box = len(lines) // len(boxes)
    for median in lines[start::lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if len(set(median.get_xdata())) == 1 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])
        if int(x) == pb2_idx:
            pb2_median = value
    return pb2_median


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs analysis of experiment output.')
    parser.add_argument('--results_dir', nargs='+',
                        type=str, help='Name of the result directory of the experiment')
    parser.add_argument('--output_dir', type=str,
                        help='Name of the output directory containing all created images.')
    parser.add_argument('--plot_type', default='portfolio', type=str,
                        help='Name of the result directory of the experiment')
    parser.add_argument('--experiment_type', default='rl', type=str, help='Specifies whether RL or DL experiment')
    parser.add_argument('--time_unit', default='timesteps_total', type=str,
                        help='timesteps (in RL) or iteration of pb2')

    args = parser.parse_args()
    assert (args.plot_type in ['portfolio', 'experiment'])
    assert (args.experiment_type in ['rl', 'dl'])

    if args.plot_type == 'experiment':
        if args.experiment_type == 'rl':
            results_path = get_src_path() / 'ray_results' / args.results_dir[0]
            output_path = get_src_path() / 'images' / f'{args.output_dir}'

            weight_dfs = load_weights(args.results_dir)
            if weight_dfs is not None:
                visualize_weights_matrix(weight_dfs.copy(), output_path)

            experiment_dataframe = load_multiple_results(args.results_dir, 'rl-eval')

            # missing_experiments(experiment_dataframe, output_path)
            normalized_score(experiment_dataframe.copy(), output_path, max_n_time_steps=15)
            critical_difference_paper(experiment_dataframe.copy(), output_path, max_n_time_steps=15)
            # training_plot_per_dataset(experiment_dataframe.copy(), output_path)
            # missing_experiments(experiment_dataframe, output_path)
            comparison_table(experiment_dataframe.copy(), output_path, n_timesteps=15)
            # hyperparameter_distribution(experiment_dataframe.copy(), str(results_path), output_path, 'hist')

            # Older results do not have runtime information
            try:
                visualize_runtime(experiment_dataframe, output_path)
            except Exception as e:
                print(e)

            comparison_table(experiment_dataframe.copy(), output_path, n_timesteps=15)
            training_plot_per_dataset(experiment_dataframe.copy(), output_path)

            visualize_all_workers(experiment_dataframe, results_path, output_path)