# Random easy/medium DeepCAD comparison

Fresh `/data/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` run,
100 GRPO steps, eight nodes with four GPUs each, single controller.
The easy and medium runs retain their separate allocations and artifacts.

Seed 42 selects 200 of the existing 512 easy tasks (score 5), plus 25
tasks at each medium score 9–16 (200 medium tasks), without replacement.
All 400 tasks are shuffled together, with no requirement for each batch
to be balanced. The in-order sampler preserves this random manifest order.
At four prompts per step, 100 steps cover the selected dataset once.
Each prompt has four generations. The existing 4,096-token output budget,
thinking setting, optimizer settings and dimensioned views are retained.

Preparation script: `configs/cad-rl/prepare-mixed.py`:

```bash
uv run --no-sync configs/cad-rl/prepare-mixed.py --data-root /data/deepcad/gym --seed 42
```

Dataset: `/data/deepcad/gym/mixed/train.dimensioned.shuffled.jsonl`.
Report: `/data/deepcad/gym/mixed/train.dimensioned.shuffled.report.json`.
Targets: `/data/deepcad/gym/mixed/references`.
Manifest SHA256: `1d33de1d326f9837f26b98dfb667469c4e5d3387d2c0a8e1a12b0f553eafda0e`.
The report records source manifest hashes and ordered per-target provenance.
All selected rows match their source tasks exactly. Copied STEP hashes,
all 400 valid positive-volume solids, all 1,200 640×640 PNGs, and native
Gym request/metadata schemas passed validation.

Recipe:
`configs/examples/recipes/vlm/vlm_grpo-nano-omni-cad-mixed-8n4g-single-controller.yaml`.
The fully resolved MasterConfig passed validation; no mixed checkpoints
existed before launch.

Persistent allocation: Slurm `7817251`, `rxm-cad-rl-zVgRJ1`.
Attach with `rxm node attach rxm-cad-rl-zVgRJ1`.
Registered rem experiment: `nano-omni-cad-mixed`.
Logs/checkpoints: `/data/deepcad/gym/runs/nano-omni-mixed`.
W&B: https://wandb.ai/nvidia/rohitkumarj-cad-rl/runs/fd5659ec3a10
Driver: `/data/deepcad/gym/runs/nano-omni-mixed/driver-1791492588.log`.

Launch once on the Ray head:

```bash
rxm node exec rxm-cad-rl-zVgRJ1 -- \
  env CAD_RL_CONFIG=configs/examples/recipes/vlm/vlm_grpo-nano-omni-cad-mixed-8n4g-single-controller.yaml \
      CAD_RL_RUN_ROOT=/data/deepcad/gym/runs/nano-omni-mixed \
  uv run --no-sync configs/cad-rl/start-driver.py
```
