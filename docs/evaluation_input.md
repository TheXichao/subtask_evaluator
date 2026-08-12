# EvaluationInput — the model-input contract

## What it is

An `EvaluationInput` (`src/subtask_checker/eval_input.py`) is the complete,
validated INPUT of one future model evaluation: everything the Qwen3-VL-4B
evaluator will receive to judge one machine-generated subtask annotation, and
nothing else. It is the boundary between the layers:

```
DATA PREPARATION   data/ + video/  →  EvaluationInput   (exists, this doc)
MODEL INFERENCE    EvaluationInput →  prompt → VLM call (future, separate module)
```

Nothing in `eval_input.py` talks to a model. The point is that the exact model
input can be inspected on disk — images and JSON side by side — and tested,
before any model is involved.

## Shape

```
EvaluationInput
├── example: TaskExample        episode identity, goal instruction, the annotated
│                               segment (start/end sec, description), video ref,
│                               provenance — the canonical model from data/schema.py
├── plan: GridSetPlan           the sampling window in chunk-video seconds, the
│                               shared badge clock, where the segment sits on it
│                               (segment_badge_start/end_sec), and the resolved
│                               GridConfig (plan.config) — from video/gridset.py
├── visual_inputs: [VisualInput]  ≥1 grid images: relative path + grid_index +
│                                 GridMeta (geometry + per-tile badge times)
└── evaluation_instruction: str   what the evaluator is asked to assess
```

No new episode/segment/timestamp representation exists here — the schema reuses
`TaskExample`, `GridSetPlan`, and `GridMeta` unchanged. Times follow the
project's established conventions: segment boundaries in seconds relative to
the episode start, badge times in seconds from the sampling-window start (the
numbers burned into the tiles), window bounds in chunk-video seconds.

`visual_inputs` is a **list** by design: one grid is the common case, but a
window longer than one grid's capacity already yields several (all on one
shared badge clock), and density experiments may later pair a coarse overview
with a dense local view. One image or many, the schema is the same.

The instruction is a development **placeholder**
(`DEFAULT_EVALUATION_INSTRUCTION`) — the real rubric and the output schema
(criteria/score/evidence/feedback) belong to the future evaluator module.
Segment times are not interpolated into it; rendering the structured fields
into a prompt is the inference module's job.

## Where the pieces come from

- **Segment metadata**: the team's pre-review `open_vocab_subtasks.jsonl`
  (machine-generated, no human review), via `build_sample.py` →
  `data/samples/dev_sample.jsonl` → `TaskExample`. It is the thing to be
  judged, not ground truth.
- **Images**: rendered by the existing grid machinery (`video/gridset.py`
  `plan_grid_set` + `build_grid_set`) from one of the candidate density
  configs (docs/grid_experiments.md); default is the proven 12-frame/336 px
  baseline. `eval_input.py` adds no second grid algorithm.
- **Model services**: role inventory (subtask_generator / reference_judge /
  evaluator) in `model_services.json`, validated by `services.py`. Endpoints
  are placeholders until real ones exist; preparation never needs them.

## Preparing and inspecting one example

```bash
uv run python scripts/prepare_input.py --index 0
uv run python scripts/prepare_input.py --example-id <id> --grid-config b_25f_tile256_2fps --scope episode
```

Output joins the example's existing directory, so everything about one segment
stays together:

```
data/samples/examples/<example_id>/
├── evaluation_input.json           the validated EvaluationInput
├── grids/<config_name>/grid_*.jpg  the images it references
└── example.json, grid.jpg, frames/   (from prepare_example.py, if run)
```

Image paths inside the JSON are relative to that directory (never absolute), so
an example directory is self-contained and portable;
`eval_input.missing_visual_files()` verifies the references resolve.
`read_evaluation_input()` loads and re-validates the JSON — that round trip is
exactly what the future inference module will consume.
