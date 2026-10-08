# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
import tarfile
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory(prefix="deepcad-arm64-rootfs-") as directory:
    with tarfile.open(
        "/home/rohitkumarj/data/deepcad/deepcad-cadquery-arm64-rootfs.tar.gz"
    ) as archive:
        archive.extractall(directory, filter="fully_trusted")
    output = Path("/home/rohitkumarj/data/deepcad/gym/cadquery-arm64-2.8.0.sqsh")
    subprocess.run(
        [
            "mksquashfs",
            directory,
            str(output),
            "-noappend",
            "-processors",
            "4",
            "-no-progress",
            "-no-xattrs",
            "-comp",
            "zstd",
        ],
        check=True,
    )
    output.replace(output.with_name("cadquery-2.8.0.sqsh"))
    print("ARM CadQuery squashfs ready", flush=True)
