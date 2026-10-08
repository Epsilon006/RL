# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path

root = Path("/home/rohitkumarj/data/deepcad/gym/enroot-runtime")
for folder in ("bin", "lib", "etc/mounts.d", "etc/hooks.d", "etc/environ.d", "config"):
    (root / folder).mkdir(parents=True, exist_ok=True)
for path in Path("/usr/bin").glob("enroot*"):
    shutil.copy2(path, root / "bin" / path.name)
shutil.copy2("/usr/bin/unsquashfs", root / "bin/unsquashfs")
shutil.copytree("/usr/lib/enroot", root / "lib", dirs_exist_ok=True)
(root / "etc/mounts.d/10-dev.fstab").write_text(
    "".join(
        f"/dev/{device} /dev/{device} none x-create=file,bind,rw,nosuid,noexec,private 0 -1\n"
        for device in ("null", "zero", "random", "urandom")
    )
)
cadlibs = root.parent / "cadquery-libs"
cadlibs.mkdir(exist_ok=True)
for name in (
    "libGL.so.1",
    "libGLX.so.0",
    "libGLdispatch.so.0",
    "libX11.so.6",
    "libxcb.so.1",
    "libXau.so.6",
    "libXdmcp.so.6",
    "libbsd.so.0",
    "libmd.so.0",
    "libXext.so.6",
    "libXrender.so.1",
):
    shutil.copy2(Path("/lib/aarch64-linux-gnu") / name, cadlibs / name)
print("Project-local Enroot runtime installed", flush=True)
