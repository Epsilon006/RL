# Medium DeepCAD comparison

Fresh `/data/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` run,
100 GRPO steps, eight nodes with four GPUs each, single controller.
The easy run has separate allocation, logs, checkpoint directory and W&B ID.

Medium means construction scores 9–16. Dataset selection takes 50 valid
training models per score, using a deterministic SHA256 permutation with
seed 42. The resulting 400 tasks are sorted by score and task ID. At four
prompts per step, 100 steps cover the whole band. Each prompt has four
generations. Input views contain dimensions in millimetres.

The preparation utility is Gym's `resources_servers/deepcad/dataset/prepare_gym_bucket.py`.
It reuses the full training-split ranking, checks source metric counts again,
and validates each exported STEP by reimporting it before rendering. Two
faulty candidates were replaced within their score. All 400 final tasks,
400 STEP references and 1,200 PNGs passed validation.

Dataset: `/data/deepcad/gym/medium/train.dimensioned.sorted.jsonl`.
Report: `/data/deepcad/gym/medium/train.dimensioned.sorted.report.json`.
Targets: `/data/deepcad/gym/medium/references`.
Logs/checkpoints: `/data/deepcad/gym/runs/nano-omni-medium`.

Recipe:
`configs/examples/recipes/vlm/vlm_grpo-nano-omni-cad-medium-8n4g-single-controller.yaml`.

Launch inside the nightly container:

```bash
CAD_RL_CONFIG=configs/examples/recipes/vlm/vlm_grpo-nano-omni-cad-medium-8n4g-single-controller.yaml \
  bash configs/cad-rl/run-cad-rl.sh
```

Persistent allocation: Slurm `7816991`, `rxm-cad-rl-KrApp4`.
Attach with `rxm node attach rxm-cad-rl-KrApp4`.
Registered rem experiment: `nano-omni-cad-medium`.

W&B: https://wandb.ai/nvidia/rohitkumarj-cad-rl/runs/17b725c5c83f
Driver: `/data/deepcad/gym/runs/nano-omni-medium/driver-1791491205.log`.

Launch validation: optimizer steps 1–6 completed with no masked samples.
Step 1 mean IoU reward was 0.124584; step 6 was 0.062329. A separate
five-task native Gym HTTP smoke check completed without infrastructure
failures. All five responses exhausted the inherited 4,096-token output
budget during reasoning before producing code, and correctly received
zero reward. These calls used the live training model, so they are a
smoke check rather than a fixed-checkpoint evaluation. The generation
budget remains identical to the easy run for this comparison.

Smoke results: `/data/deepcad/gym/medium/nano-smoke-summary.json` and
`/data/deepcad/gym/medium/nano-smoke-rollouts.jsonl`.
