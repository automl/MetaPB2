"""
Example how to use the portfolio initializer with Ray Tune.
This works with all PB2 variants introduced in the paper.
"""

from pathlib import Path

from ray import air, tune
from ray.tune import sample_from

from src.algorithms.algorithm_utils import sampler
from src.algorithms.log_pb2 import LogPB2
from src.portfolio_pb2 import read_portfolio
from src.rl.env_classes import register_carl_env
from src.utils import get_float_transformer


def main():
    scheduler = LogPB2(
        time_attr="timesteps_total",
        perturbation_interval=1_000,
        hyperparam_bounds={
            "lambda": [0.9, 0.99],
            "clip_param": [0.1, 0.5],
            "lr": [1e-5, 1e-3],
        },
        custom_explore_fn = get_float_transformer(["lr", "lambda", "clip_param"]), # ValueError fix
    )

    env_name = "mountain_car"
    register_carl_env(env_name) # This is only necessary because the carl environments are not registered by default
    portfolio = read_portfolio(Path(__file__).parents[1] / "meta_data" / "example" / "portfolio.json")
    param_space = {
        "lr": sample_from(sampler(portfolio, "lr")),
        "lambda": sample_from(sampler(portfolio, "lambda")),
        "clip_param": sample_from(sampler(portfolio, "clip_param")),
        "env_config": {'g_factor': 2.},
        "train_batch_size": 1_000,
        "env": env_name,
        "framework": "torch",
        "num_workers": 0,
    }
    run_config = air.RunConfig(stop={"timesteps_total": 5_000})
    tune_config = tune.TuneConfig(metric="episode_reward_mean",
                                  mode="max",
                                  scheduler=scheduler,
                                  num_samples=4)
    tuner = tune.Tuner(
        "PPO",
        run_config=run_config,
        tune_config=tune_config,
        param_space=param_space,
    )
    result_grid = tuner.fit()
    print("Training Finished.")
    print(result_grid)

if __name__ == "__main__":
    main()