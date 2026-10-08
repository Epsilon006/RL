# CAD-RL on HSG

Worktree: `/home/rohitkumarj/code/RL/nemo-rl-cad-rl`, branch `rohit/cad-rl`.
Base: `nv/rohit/unified-teacher-supervl3p5`, commit `3fc5ea315d3f11bee73fe5b7d41cda74905443b7`.
Gym environment: `3rdparty/Gym-workspace/Gym/resources_servers/deepcad`.

The model sees dimensioned front, side, and top orthographic images and generates Python with CadQuery.
Labels give overall dimensions, circle/arc radii, and rectangular additive extrusion depths in mm.
Reward is solid-volume IoU after similarity Procrustes alignment. Invalid code/geometry earns zero;
infrastructure failures are masked. Generated code runs in a fresh ARM CadQuery Enroot container,
without the reference directory or inherited Gym configuration.

All 161,240 training shapes are ranked in `/data/deepcad/gym/train.sorted.difficulty.jsonl`.
The run uses the easiest 512 rendered tasks in `/data/deepcad/gym/train.dimensioned.sorted.jsonl`.
Their score is 5; the full split ranges from 5 to 102. Higher scores begin at index 13,559.
Expand the rendering limit before a longer curriculum run. Validation examples are selected after
ranking all 8,946 validation tasks. Preparation reports reconstruction failures.

Difficulty is `4 * extrusions + sketch_curves + 2 * additional_loops + 4 * cuts_and_intersections`,
computed only from profiles referenced by extrusion features. Ties use task ID.
The run disables shuffling and uses the single-controller `in_order` sampler.

Experiment: `nano-omni-cad-difficulty`, model
`/data/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`.
The first validation run completed 20 steps. Its checkpoint was deleted at the user's request.
The fresh dimensioned-view run starts from the original model and has 100 steps,
4 prompts per step, and 4 generations per prompt (400 training tasks, 1,600 responses).
Eight nodes have four GPUs each: four training nodes and four generation nodes.
Two-node training segments match the allocated NVLink domains.

Inside the nightly container, launch with:

```bash
bash configs/cad-rl/run-cad-rl.sh
```

The script sets the project-local Enroot runtime and runs:

```bash
uv run --no-sync examples/run_grpo_single_controller.py \
  --config configs/examples/recipes/vlm/vlm_grpo-nano-omni-cad-difficulty-8n4g-single-controller.yaml
```

The parent image is `/home/rohitkumarj/data/enroot-containers/rl.nightly.sep30.2026.sqsh`.
The CadQuery sandbox image is `/data/deepcad/gym/cadquery-2.8.0.sqsh`.
Gym server virtual environments are in `/data/deepcad/gym/venvs`; Ray is pinned to 2.56.1
to match the nightly training cluster. Enroot writable files stay under `/tmp/deepcad-enroot`.

Logs and checkpoints: `/data/deepcad/gym/runs/nano-omni-difficulty`.
`start-driver.py` writes a timestamped driver log and an exit marker.
`sandbox-smoke.py` verifies real execution, reference/config isolation, continuous IoU, and invalid-code zero.

Persistent allocation: Slurm `7815932`, handle `rxm-cad-rl-WR5ScC`, four-hour limit.
Attach from the laptop with `rxm node attach rxm-cad-rl-WR5ScC`.

Fresh 100-step dimensioned run W&B: https://wandb.ai/nvidia/rohitkumarj-cad-rl/runs/93d24a5a5f72
Driver: `/data/deepcad/gym/runs/nano-omni-difficulty/driver-1791489617.log`.

Validation: 9 source-preparation tests passed; all 512 tasks passed native Gym schema
checks with identical target metadata/order; five dimensioned example tasks passed Gym collation.

The independent medium comparison uses scores 9–16, 400 tasks and 100 steps.
See `MEDIUM-RUN.md` for dataset validation, allocation and launch details.

The third comparison uses a seed-42 shuffled 50/50 easy/medium mix:
400 tasks and 100 steps. See `MIXED-RUN.md` for provenance and run details.
