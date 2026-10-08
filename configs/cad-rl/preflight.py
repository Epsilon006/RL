# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

os.environ.setdefault("CAD_RL_RUN_ID", "preflight")
os.environ.setdefault("CAD_RL_CAPTURE_TOKEN", "preflight")
register_omegaconf_resolvers()
config = MasterConfig(
    **OmegaConf.to_container(
        load_config(
            "configs/examples/recipes/vlm/vlm_grpo-nano-omni-cad-difficulty-8n4g-single-controller.yaml"
        ),
        resolve=True,
    )
)
print(
    "VALID CONFIG", config.cluster, config.async_rl.sampler, config.policy["model_name"]
)
print(
    "Gym dependencies",
    Path("/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python").exists(),
)
