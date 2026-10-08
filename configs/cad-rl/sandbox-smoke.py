# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import os
from pathlib import Path

import nemo_gym.global_config as global_config
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from omegaconf import DictConfig
from resources_servers.deepcad.app import (
    DeepCADResourcesServer,
    DeepCADResourcesServerConfig,
)

root = Path("/data/deepcad/gym")
os.environ.update(
    {
        "PATH": str(root / "enroot-runtime/bin") + ":" + os.environ["PATH"],
        "ENROOT_LIBRARY_PATH": str(root / "enroot-runtime/lib"),
        "ENROOT_SYSCONF_PATH": str(root / "enroot-runtime/etc"),
        "ENROOT_CONFIG_PATH": str(root / "enroot-runtime/config"),
        "ENROOT_ALLOW_SUPERUSER": "yes",
        "ENROOT_MOUNT_HOME": "no",
        "ENROOT_MAX_PROCESSORS": "4",
        "ENROOT_DATA_PATH": "/tmp/deepcad-enroot/data",
        "ENROOT_CACHE_PATH": "/tmp/deepcad-enroot/cache",
        "ENROOT_RUNTIME_PATH": "/tmp/deepcad-enroot/runtime",
        "NEMO_GYM_CONFIG_DICT": "secret-fixture-do-not-inherit",
    }
)
global_config._GLOBAL_CONFIG_DICT = DictConfig({})
client = ServerClient(
    head_server_config=BaseServerConfig(host="127.0.0.1", port=18800),
    global_config_dict=DictConfig({}),
)
server = DeepCADResourcesServer(
    config=DeepCADResourcesServerConfig(
        host="127.0.0.1",
        port=18801,
        name="deepcad",
        entrypoint="app.py",
        reference_root=str(root / "references"),
        sandbox_image=str(root / "cadquery-2.8.0.sqsh"),
        sandbox_python="/usr/local/bin/python",
        sandbox_provider={
            "enroot": {
                "create": {"rw": True, "remap_root": False, "bypass_entrypoint": False}
            }
        },
    ),
    server_client=client,
)


async def main():
    row = json.loads((root / "train.sorted.jsonl").read_text().splitlines()[0])
    reference = row["verifier_metadata"]["reference_step"]
    code = "import os, cadquery as cq\nassert not os.environ.get('NEMO_GYM_CONFIG_DICT')\nassert not os.path.exists('/data/deepcad/gym/references')\nresult = cq.Workplane('XY').box(1,2,3)"
    good = await server._evaluate(code, reference)
    bad = await server._evaluate(
        "raise RuntimeError('invalid generated code')", reference
    )
    (root / "sandbox-smoke.json").write_text(
        json.dumps({"valid": good, "invalid": bad}, indent=2)
    )
    print(json.dumps({"valid": good, "invalid": bad}), flush=True)
    assert not good.get("mask_sample") and 0 < good["reward"] <= 1
    assert not bad.get("mask_sample") and bad["reward"] == 0


asyncio.run(main())
