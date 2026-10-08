# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import time
from pathlib import Path

root = Path(
    os.environ.get("CAD_RL_RUN_ROOT", "/data/deepcad/gym/runs/nano-omni-difficulty")
)
root.mkdir(parents=True, exist_ok=True)
started = int(time.time())
log = root / f"driver-{started}.log"
print(f"Driver log: {log}", flush=True)
with log.open("w") as output:
    result = subprocess.run(
        ["bash", "/opt/nemo-rl/configs/cad-rl/run-cad-rl.sh"],
        stdout=output,
        stderr=subprocess.STDOUT,
    )
(root / f"driver-{started}.exit.json").write_text(
    json.dumps({"exit_code": result.returncode, "log": str(log)})
)
print("Driver exit:", result.returncode, flush=True)
raise SystemExit(result.returncode)
