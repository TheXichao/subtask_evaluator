# CLAUDE.md — Subtask Checker

## What this project is

Groundwork for a video-based evaluator built on a Qwen3-VL 4B-class VLM. Eventually:

```text
video + instruction + one subtask (start/end/description)
                    ↓
             Qwen3-VL evaluator
                    ↓
criteria + score + evidence + feedback (structured JSON)
```

The model should judge whether a video segment was correctly labelled/performed
(boundaries, correct action, failed/reattempted actions, partial completion),
with evidence grounded in the video.

**Current phase: input-side data pipeline only.** The pipeline below works and is
tested. No evaluation rubric, no training labels, no fine-tuning yet — do not
build them unless explicitly asked.

```text
team pre-review subtask JSONL → loader → canonical TaskExample
    → video segment → sampled frames → contact-sheet grid → inspect
    → official Qwen SFT record (placeholder targets until real labels exist)
```

## Hard rules

1. **Read-only external directories** — never modify, rename, delete, move, or
   write anything under:
   - `/home/xichao/remote/410_g1_ir` (LeRobot v3 dataset, sshfs mount)
   - `/home/xichao/remote/Unitree_Robo_Describer` (team subtasker repo, sshfs mount)
   If a task seems to require modifying them, stop and ask.
2. **No human-labelled data at this stage.** Only the machine-generated,
   pre-review annotations (`open_vocab_subtasks.jsonl`) are used. Never read the
   human-edited `open_vocab_subtasks_results_normalized_for_review.json`, review
   diffs, viewer verdicts, or anything derived from human review. Do not
   silently mix supervision sources — `Provenance.annotation_source` exists so
   that can never happen.
3. **No training work yet**: no Qwen3-VL downloads, LoRA, MS-SWIFT/LLaMA-Factory
   configs, distributed setup, or large-scale dataset generation. Those are
   external tools for a later stage, never core architecture of this repo.
4. **Generated subtasks are not ground truth.** They are the thing to be judged
   (team's own time-IoU vs human review: 0.83–0.98 per task). Never name them
   `ground_truth`; use `source_annotation` / `generated_annotation` /
   `gold_label` etc. as appropriate.
5. **Don't process the full dataset** when a small sample answers the question.
   Dev sample ≈ 10–50 subtasks.
6. **Don't commit large files**: videos, frames, grids, checkpoints. `data/` is
   gitignored; generated artifacts are reproducible from the mount.

## Verified data facts (details: docs/dataset.md)

- Source of truth: `<task>/meta/motion_desc_pipeline/open_vocab_subtasks.jsonl`
  per task — one row per machine-generated subtask. 34 task dirs, 16,774 rows.
  (`TEST-clear_plates` and `Motor_STAGE_*` are excluded non-task dirs.)
- Each row carries: `goal` (episode instruction), `episode_id`/`episode_index`,
  `step_index`, `subtask` (description), `start_time`/`end_time` (**seconds
  relative to episode start**, 1 s resolution from MM:SS), `video_key`,
  `source_chunk_video`, `source_from_timestamp`/`source_to_timestamp`.
- **Deterministic segment mapping** (verified against footage):
  `chunk_video_time = source_from_timestamp + episode_time`. The JSONL is
  self-sufficient for the head camera; no parquet lookups needed.
- **Camera trap:** LeRobot v3 splits chunk mp4s by *size*, so per-camera
  file/offset differ for the same episode. `source_chunk_video` +
  `source_from_timestamp` are valid ONLY for the row's own `video_key`
  (always `observation.images.head_stereo_left`). Adding wrist cameras requires
  resolving through `meta/episodes/*.parquet` (working reference:
  `~/Dev/evaluation/code/video_index.py`). Never path-swap camera directories.
- Team JSON stores server paths (`/mnt/raid0/supeng_g1_data/410_g1_ir/...`);
  translate via `config.resolve_server_path`. Mount override:
  `SUBTASK_CHECKER_DATA_ROOT` (default `~/remote/410_g1_ir`).
- Videos: AV1 640×480 30 fps. Use ffmpeg with `-ss` before `-i` (OpenCV often
  fails on these streams); ffmpeg can exit 0 with an empty file on
  past-the-end seeks, so check JPEG magic bytes.

## Architecture

```text
src/subtask_checker/
  config.py        canonical shared paths (ProjectPaths/PATHS, env-overridable),
                   server→local path translation, require_writable guard
  experiments.py   ExperimentConfig: validated experiments/*.json override files
  data/source.py   team JSONL format (SourceSubtaskRow) — their field names stop here
  data/schema.py   canonical models: TaskExample, Segment, VideoRef, Provenance
  data/prepare.py  source row → TaskExample (the ONE adaptation boundary),
                   validation, deterministic stratified sampling
  data/episodes.py task → episode → subtask hierarchy: EpisodeSubtasks groups
                   TaskExamples per episode, validating the verified invariants
                   (one goal, one video window, unique ascending steps)
  video/frames.py  window computation (±3s pad / full episode, episode-clamped),
                   frame planning (≤12 frames, ≥0.5s interval, badge times
                   rebased to window start), ffmpeg extraction + ffprobe
  video/grid.py    contact sheet: 4 cols, 336px tiles, burned-in time badges (Pillow)
  video/gridset.py configurable density experiments: GridConfig (rows×cols, fps,
                   tile px) → multi-grid plan + GridSetMetrics (token estimates);
                   see docs/grid_experiments.md
  eval_input.py    EvaluationInput: the data→model input contract (TaskExample +
                   GridSetPlan + ≥1 grid image refs + instruction). Assembles via
                   video/gridset — no second grid algorithm, no model calls. The
                   inference layer is a future SEPARATE module consuming this.
  services.py      model-service roles (subtask_generator/reference_judge/evaluator)
                   from model_services.json — endpoints stay null placeholders until
                   real ones exist (server ports are behind SSH tunnels, not URLs)
  qwen_export.py   EvaluationInput → Qwen SFT record
                   ({"image": […], "conversations": [{"from": "human"/"gpt"}]})
                   + self-contained dataset dir with provenance manifest. One
                   record format serves BOTH trainers: LLaMA-Factory ShareGPT
                   (preferred; needs media_dir + image_max_pixels in the YAML)
                   and official qwen-vl-finetune. Assistant turns need an
                   explicit QwenTarget with a source tag; PLACEHOLDER_TARGET
                   until real labels exist (docs/qwen_dataset.md)
scripts/           thin CLIs only; logic lives in src/
experiments/       source-controlled experiment override files (JSON)
tests/             deterministic tests (parsing, validation, mapping, planning, grid)
docs/dataset.md    verified source schema; distinguishes confirmed/assumed/unresolved
docs/qwen_dataset.md   verified official Qwen SFT format + export design
docs/investigation_report.md   full 2026-08-11 background investigation
```

Data contracts are Pydantic end to end; no raw dicts across stage boundaries.
Nothing downstream of `data/prepare.py` may touch the team's field names.

Deliberately absent (add only when a real requirement appears): `labels/`,
model wrappers, caching layers, notebooks. Avoid
`BaseModel`/`ModelFactory`/`PipelineManager`-style speculative abstractions
and a generic `utils.py`.

## Configuration (details: docs/configuration.md)

Component-oriented, never one monolithic global config (the old evaluation
project's single `config.py` is the anti-pattern here):

- **Shared environment** has ONE canonical source: `config.py::PATHS`
  (frozen Pydantic `ProjectPaths`, `SUBTASK_CHECKER_*` env overrides) — repo,
  read-only mounts, evaluation-reference root, generated-output root only.
  `require_writable()` guards artifact writes against the read-only roots.
- **Component settings** live in the component as Pydantic models with field
  defaults (e.g. `video/gridset.py::GridConfig`); functions receive config
  explicitly, never via hidden global state.
- **Experiments** override component defaults via source-controlled
  `experiments/*.json` (validated by `experiments.py::ExperimentConfig`,
  `extra="forbid"`); artifacts record the experiment name + resolved configs
  so runs reproduce from the repo alone. Never hard-code experimental choices
  into defaults, and never vary settings by editing source.
- **Not configuration**: dataset-layout facts (server prefix, JSONL relpath),
  evaluation rubric/judgement logic, data schemas — those are code.
- New component → new config model in that component + optional field on
  `ExperimentConfig` when experiments need it. No registry/discovery/inheritance.

## Commands

```bash
uv sync                                              # needs ffmpeg on PATH (AV1)
uv run python scripts/build_sample.py --count 20     # deterministic dev sample (seed 42)
uv run python scripts/prepare_example.py --index 0   # or --example-id <id>
uv run python scripts/compare_grid_configs.py --index 0 [--scope episode]  # density candidates side by side
uv run python scripts/prepare_input.py --index 0     # full model input for one example (no model call)
uv run python scripts/export_qwen_dataset.py --name dev20   # official Qwen SFT dataset (placeholder targets)
uv run pytest
```

`prepare_example.py` writes `data/samples/examples/<id>/` with `example.json`
(canonical example + frame plan + grid metadata), `grid.jpg`, and `frames/` —
this is exactly what a future evaluator would receive, inspectable.

## Prompt log (required)

After every user prompt, before ending the turn, append an entry to
`docs/prompt_log.md` (create if missing; newest at the bottom, never rewrite
old entries):

```markdown
## 2026-08-11 16:26 — <3–6 word topic>
- asked: <one short bullet — what the user requested>
- done: <1–3 short bullets — what actually happened>
```

- Date+time from `date '+%Y-%m-%d %H:%M'`; bullets telegraphic, no prose.
- Report outcomes honestly: failures, partial work, and "nothing changed" count.

## Frame-grid conventions (adapted from ~/Dev/evaluation, proven there)

- Window = segment ±3 s context (or the whole episode, for experiments),
  **clamped to the episode's own span in the chunk file** (chunks concatenate
  episodes; an unclamped pad shows the neighbouring episode).
- Time badges are burned into pixels, counting from the window start — the
  earlier project proved models fabricate boundary times without this. In a
  multi-grid set, all grids share the window clock (no per-grid reset).
- **Density is an explicit design question, not a constant.** Target: useful
  temporal information per unit of visual context on a 4B VLM — never optimise
  frame count or image quality alone, and never use JPEG size as a proxy for
  context cost. Candidate configs (rows×cols, fps, tile px) live in
  `video/gridset.py`; compare them empirically with
  `scripts/compare_grid_configs.py`; findings in docs/grid_experiments.md.
- `prepare_example.py` defaults stay at the proven geometry (≤12 frames,
  ≥0.5 s interval, 336 px tiles — safe on shared vLLM engines) until the
  experiments justify changing them.
- Uniform sampling is the deliberate baseline; boundary-/redundancy-aware
  sampling only when an experiment shows uniform is insufficient.
- `~/Dev/evaluation` remains a source of reference implementations and lessons
  (judge schemas, calibration, synthetic corruptions, camera resolution) — adapt
  pieces deliberately; never copy the project or adopt its architecture wholesale.

## How to make decisions

1. Inspect actual data/code before implementing against it; don't trust
   filenames or memory — this file records only what was verified.
2. Prefer the simplest solution compatible with the actual data; boring and
   conventional beats clever.
3. Preserve provenance on everything (source file, row index, generator model).
4. If source data is ambiguous or a mapping can't be established reliably,
   stop and document — never invent a mapping.
5. Keep every transformation independently runnable and inspectable.
6. Small, testable changes; after meaningful changes report what changed, how to
   run it, what was verified, and any unresolved assumptions.

## Future direction (context, not a to-do list)

When the input pipeline has proven itself: baseline Qwen3-VL inference → define
rubric/output schema → construct trustworthy supervision (the hard part — see
docs/investigation_report.md for why prompt-only evaluation hit a ceiling and
why labels, not models, are the bottleneck) → training dataset → LoRA fine-tune
→ held-out evaluation. Judge the evaluator on task-level quality (criterion
accuracy, boundary quality, evidence grounding, agreement with reliable human
judgements), not training loss. Do not start any of this unprompted.
