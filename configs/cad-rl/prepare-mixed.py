# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select and shuffle existing dimensioned easy/medium tasks for a comparison."""

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    root = args.data_root
    output = root / "mixed"
    manifest = output / "train.dimensioned.shuffled.jsonl"
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest}")
    sources = [
        root / "train.dimensioned.sorted.jsonl",
        root / "medium/train.dimensioned.sorted.jsonl",
    ]
    easy, medium = [
        [json.loads(line) for line in source.read_text().splitlines()]
        for source in sources
    ]
    assert len(easy) >= 200 and all(
        row["verifier_metadata"]["difficulty"] == 5 for row in easy
    )
    assert {row["verifier_metadata"]["difficulty"] for row in medium} == set(
        range(9, 17)
    )
    # Sample each medium score equally, then shuffle all selected prompts together.
    rng = random.Random(args.seed)
    selected = [(row, root / "references", "easy") for row in rng.sample(easy, 200)]
    for score in range(9, 17):
        candidates = [
            row for row in medium if row["verifier_metadata"]["difficulty"] == score
        ]
        selected.extend(
            (row, root / "medium/references", "medium")
            for row in rng.sample(candidates, 25)
        )
    rng.shuffle(selected)
    assert len({row["verifier_metadata"]["task_id"] for row, _, _ in selected}) == 400
    references = output / "references"
    references.mkdir(parents=True, exist_ok=True)
    provenance = []
    for row, reference_root, band in selected:
        metadata = row["verifier_metadata"]
        name = Path(metadata["reference_step"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Unsafe reference path: {name}")
        source = reference_root / name
        target = references / name
        if not source.is_file() or not source.stat().st_size:
            raise ValueError(f"Missing reference: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        provenance.append(
            {
                "task_id": metadata["task_id"],
                "band": band,
                "difficulty": metadata["difficulty"],
                "source_reference": str(source),
                "reference_sha256": digest,
            }
        )
    payload = "".join(json.dumps(row) + "\n" for row, _, _ in selected)
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(payload)
    report = {
        "seed": args.seed,
        "selection": "200 easy; 25 per medium score; random shuffle",
        "tasks": 400,
        "band_counts": dict(Counter(row["band"] for row in provenance)),
        "difficulty_counts": dict(Counter(row["difficulty"] for row in provenance)),
        "manifest_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "sources": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sources
        ],
        "ordered_tasks": provenance,
    }
    manifest.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(manifest)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "seed",
                    "tasks",
                    "band_counts",
                    "difficulty_counts",
                    "manifest_sha256",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
