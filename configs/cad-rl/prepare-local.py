# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

data = Path("/home/rohitkumarj/data/deepcad")
repo = Path("/home/rohitkumarj/code/RL/nemo-rl-cad-rl")
gym = repo / "3rdparty/Gym-workspace/Gym"
python = gym / ".venv/bin/python"
with tempfile.TemporaryDirectory(prefix="deepcad-data-") as directory:
    root = Path(directory)
    archive = root / "cad_json.tar.gz"
    shutil.copyfile(data / archive.name, archive)
    with tarfile.open(archive) as source:
        source.extractall(root, filter="fully_trusted")
    shutil.copyfile(
        data / "train_val_test_split.json", root / "train_val_test_split.json"
    )
    print("Node-local DeepCAD JSON ready", flush=True)
    for split, limit, output, refs in (
        (
            "train",
            512,
            data / "gym/train.dimensioned.sorted.jsonl",
            data / "gym/references",
        ),
        (
            "validation",
            5,
            repo
            / "3rdparty/Gym-workspace/Gym/resources_servers/deepcad/data/example.jsonl",
            repo / "3rdparty/Gym-workspace/Gym/resources_servers/deepcad/data",
        ),
    ):
        subprocess.run(
            [
                str(python),
                "-m",
                "resources_servers.deepcad.dataset.prepare_gym",
                "--data-root",
                str(root),
                "--split",
                split,
                "--limit",
                str(limit),
                "--output",
                str(output),
                "--reference-root",
                str(refs),
            ],
            cwd=gym,
            check=True,
        )
    print("Difficulty-sorted train and example sets ready", flush=True)
