# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate embedded views while keeping verified STEP targets and task order."""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cadquery as cq


def refresh(arguments):
    row, source_root, data_root, reference_root, views_root = arguments
    sys.path.insert(0, str(source_root))
    # The selected Gym checkout is resolved separately in each preparation worker.
    from resources_servers.deepcad.dataset.prepare_gym import render_views, task_row

    metadata = row["verifier_metadata"]
    identifier = metadata["task_id"]
    document = json.loads((data_root / "cad_json" / f"{identifier}.json").read_text())
    shape = cq.importers.importStep(
        str(reference_root / metadata["reference_step"])
    ).val()
    images = render_views(shape, views_root / identifier, document=document)
    metrics = {**metadata["construction_metrics"], "difficulty": metadata["difficulty"]}
    result = task_row(identifier, metadata["reference_step"], images, metrics=metrics)
    result["verifier_metadata"] = metadata
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data-root",
        "reference-root",
        "views-root",
        "input",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "3rdparty/Gym-workspace/Gym",
        help="Gym checkout containing the DeepCAD preparation module",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    arguments = [
        (row, args.source_root, args.data_root, args.reference_root, args.views_root)
        for row in rows
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with (
        ProcessPoolExecutor(max_workers=args.workers) as pool,
        temporary.open("w") as output,
    ):
        for index, row in enumerate(pool.map(refresh, arguments), 1):
            output.write(json.dumps(row) + "\n")
            if index % 32 == 0 or index == len(rows):
                print(json.dumps({"rendered": index, "total": len(rows)}), flush=True)
    temporary.replace(args.output)
    print(
        json.dumps(
            {"output": str(args.output), "tasks": len(rows), "views": 3 * len(rows)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
