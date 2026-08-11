# subtask_checker

Groundwork for fine-tuning a Qwen3-VL-4B evaluator of robot-video subtask
annotations. **Current scope: the input-side data pipeline only** — take the
team's automatically generated subtask annotations, resolve one subtask to its
exact video segment, and produce an inspectable visual representation. No
evaluation labels, no training, no human-reviewed data yet.

```
Existing team data          <task>/meta/motion_desc_pipeline/open_vocab_subtasks.jsonl
       ↓                    (pre-review subtasker output, read-only mount)
Subtask loader              src/subtask_checker/data/source.py
       ↓
Small sample                scripts/build_sample.py → data/samples/dev_sample.jsonl
       ↓
Canonical TaskExample       src/subtask_checker/data/schema.py
       ↓
Video segment               chunk_time = source_from_timestamp + episode_time
       ↓
Frames / grid               scripts/prepare_example.py → frames + contact sheet
       ↓
[future: evaluator]         {criteria, score, evidence, feedback} — not built yet
       ↓
[future: fine-tuning]       Qwen3-VL-4B via MS-SWIFT / LLaMA-Factory — not built yet
```

## Setup

```bash
uv sync
```

Needs `ffmpeg` on PATH (with AV1 decode; stock Ubuntu ffmpeg works) and
the read-only dataset mount. Canonical paths live in
`src/subtask_checker/config.py` (`PATHS`), each overridable by environment
variable:

```bash
export SUBTASK_CHECKER_DATA_ROOT=~/remote/410_g1_ir    # dataset mount (default)
# also: SUBTASK_CHECKER_TEAM_REPO_ROOT, SUBTASK_CHECKER_EVALUATION_ROOT,
#       SUBTASK_CHECKER_OUTPUT_ROOT
```

The external directories (`~/remote/410_g1_ir`, `~/remote/Unitree_Robo_Describer`)
are team-owned and treated as strictly read-only — enforced at the write sites by
`config.require_writable`. Nothing is copied into the repo and generated
artifacts under `data/` are gitignored. Configuration conventions (shared paths
vs component config vs experiment overrides): see `docs/configuration.md`.

## Commands

```bash
# 1. Select a small deterministic development sample (one subtask per task, seed 42)
uv run python scripts/build_sample.py --count 20

# 2. Turn one sampled subtask into a self-contained inspectable example
uv run python scripts/prepare_example.py --index 0
uv run python scripts/prepare_example.py --example-id 410_g1_ir/clear_plates/episode_000617/step_005

# 3. Compare frame/grid densities for one example (see docs/grid_experiments.md)
uv run python scripts/compare_grid_configs.py --index 0 --scope episode
uv run python scripts/compare_grid_configs.py --index 0 --experiment experiments/tile_floor.json

# 4. Tests
uv run pytest
```

`prepare_example.py` writes `data/samples/examples/<example_id>/` containing
`example.json` (canonical example + frame plan + grid metadata), `grid.jpg`
(contact sheet: ≤12 tiles, 336 px wide, time badges burned in, ±3 s context
around the segment), and `frames/` (the individual sampled frames).

## Layout

```
src/subtask_checker/
  config.py          canonical shared paths (PATHS, env-overridable) +
                     server→local path translation + read-only write guard
  experiments.py     validated experiment override files (ExperimentConfig)
  data/source.py     the team's JSONL format (their field names stop here)
  data/schema.py     canonical models: TaskExample, Segment, VideoRef, Provenance
  data/prepare.py    source row → TaskExample, deterministic stratified sampling
  video/frames.py    window computation, frame planning, ffmpeg extraction/probe
  video/grid.py      contact-sheet JPEG with burned-in time badges (Pillow)
  video/gridset.py   configurable grid-density experiments (GridConfig, metrics)
scripts/             thin CLIs over src/ (+ dataset_stats.py, a read-only stats sweep)
experiments/         source-controlled experiment definitions (JSON)
docs/configuration.md          configuration conventions
docs/dataset.md                the verified source schema and video mapping
docs/grid_experiments.md       grid-density experiment design + findings
docs/investigation_report.md   full background investigation (2026-08-11)
```

Design notes: the team's JSONL is adapted to the canonical model at exactly one
boundary (`data/prepare.py::to_task_example`); everything downstream sees only
Pydantic models. Frame-grid geometry (pad, tile budget, badge protocol) follows
the configuration proven in the earlier `~/Dev/evaluation` work, reimplemented
minimally. Only the head camera is used — per-camera chunk offsets differ (see
docs/dataset.md, "camera trap") and the head camera is self-describing in the
source rows.
