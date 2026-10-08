#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd /opt/nemo-rl
export RAY_TMPDIR=/tmp
export PATH="/data/deepcad/gym/enroot-runtime/bin:$PATH"
export ENROOT_LIBRARY_PATH=/data/deepcad/gym/enroot-runtime/lib
export ENROOT_SYSCONF_PATH=/data/deepcad/gym/enroot-runtime/etc
export ENROOT_CONFIG_PATH=/data/deepcad/gym/enroot-runtime/config
export ENROOT_ALLOW_SUPERUSER=yes
export ENROOT_MOUNT_HOME=no
export ENROOT_MAX_PROCESSORS=4
export LD_LIBRARY_PATH="/data/deepcad/gym/cadquery-libs:${LD_LIBRARY_PATH:-}"
export CAD_RL_RUN_ID="${CAD_RL_RUN_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex[:12])')}"
export CAD_RL_CAPTURE_TOKEN="${CAD_RL_CAPTURE_TOKEN:-$(python -c 'import secrets; print(secrets.token_hex(32))')}"
exec uv run --no-sync examples/run_grpo_single_controller.py \
  --config "${CAD_RL_CONFIG:-configs/examples/recipes/vlm/vlm_grpo-nano-omni-cad-difficulty-8n4g-single-controller.yaml}" "$@"
