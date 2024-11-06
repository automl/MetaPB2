"""
Modified version of https://github.com/automl/CARL/blob/main/carl/envs/brax/carl_inverted_pendulum.py
"""
from __future__ import annotations

import numpy as np

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from src.rl.brax_env.carl_brax_env_fix import CARLBraxEnvFix


class CARLBraxInvertedPendulumFix(CARLBraxEnvFix):
    env_name: str = "inverted_pendulum"
    asset_path: str = "envs/assets/inverted_pendulum.xml"

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "gravity": UniformFloatContextFeature(
                "gravity", lower=-1000, upper=-1e-6, default_value=-9.8
            ),
            "friction": UniformFloatContextFeature(
                "friction", lower=0, upper=100, default_value=1
            ),
            "elasticity": UniformFloatContextFeature(
                "elasticity", lower=0, upper=100, default_value=0
            ),
            "mass_cart": UniformFloatContextFeature(
                "mass_cart", lower=1e-6, upper=np.inf, default_value=1
            ),
            "mass_pole": UniformFloatContextFeature(
                "mass_pole", lower=1e-6, upper=np.inf, default_value=1
            ),
            "ang_damping": UniformFloatContextFeature(
                "ang_damping", lower=-np.inf, upper=np.inf, default_value=-0.05
            ),
            "viscosity": UniformFloatContextFeature(
                "viscosity", lower=0, upper=np.inf, default_value=0
            ),
        }
