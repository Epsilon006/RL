#!/usr/bin/env bash
set -euo pipefail
cd /opt/nemo-rl/3rdparty/Gym-workspace/Gym
root=/data/deepcad/gym/venvs
uv venv --python /opt/nemo_rl_venv/bin/python "$root/common"
UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_INSTALLS=2 uv pip install --python "$root/common/bin/python" -e '.[sandbox]' cadquery==2.8.0 scipy numpy ray==2.56.1
for server in resources_servers/deepcad responses_api_agents/simple_agent responses_api_models/vllm_model; do
  mkdir -p "$root/$server"
  ln -s "$root/common" "$root/$server/.venv"
done
"$root/common/bin/python" -c 'import cadquery,nemo_gym; print("Gym runtime ready", cadquery.__version__)'
