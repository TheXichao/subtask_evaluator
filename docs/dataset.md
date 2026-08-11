# The team's subtask data, as this project reads it

Everything here was verified against the files and code on 2026-08-11 unless marked
otherwise. The full background investigation (subtasker pipeline, review process,
prior evaluation work) is in [investigation_report.md](investigation_report.md);
per-task statistics are in [dataset_stats.json](dataset_stats.json)
(regenerate with `scripts/dataset_stats.py`).

## Where the data lives

Read-only sshfs mount (configurable via `SUBTASK_CHECKER_DATA_ROOT`):

```
/home/xichao/remote/410_g1_ir            # unitree@10.0.8.204:/mnt/raid0/supeng_g1_data/410_g1_ir
```

LeRobot **v3.0** datasets, one directory per manipulation task (34 task dirs with
subtasker output; `TEST-clear_plates` is a test copy and `Motor_STAGE_*` are
unrelated experiments — both excluded). Per task:

```
<task>/
  videos/<video_key>/chunk-000/file-NNN.mp4      # AV1 640x480 30fps; EPISODES CONCATENATED by size
  meta/episodes/chunk-000/file-000.parquet       # per-episode, per-camera chunk/file/from_ts/to_ts
  meta/tasks.parquet                             # instruction sentence(s)
  meta/downsample.json                           # the ~100 episodes selected for annotation
  meta/motion_desc_pipeline/
    open_vocab_subtasks.jsonl                    # <-- WHAT THIS PROJECT READS (pre-review)
    open_vocab_subtasks_results_normalized_for_review.json   # human-edited — NOT used
    annotation_by_episode.json, ...              # downstream deliverables — NOT used
```

## The file this project reads

`meta/motion_desc_pipeline/open_vocab_subtasks.jsonl` — one JSON object per line,
one line per generated subtask. Produced by the team's subtasker (Qwen3.6-27B-sft
over the whole 2 Hz / 320×240 episode video) followed by rule normalization
(`rule_canonical_v2`). Written **before** human review: for clear_plates the JSONL
has 499 rows while the human-reviewed copy has 491 — the JSONL retains the
machine output. 16,774 rows across all 34 tasks.

Representative row (clear_plates, abridged — normalization bookkeeping fields omitted):

```json
{
  "dataset": "410_g1_ir",
  "task": "clear_plates",
  "goal": "Place the plates on the rack from right to left in the order of red, green, and blue.",
  "episode_id": "episode_000017",
  "episode_index": 17,
  "step_index": 0,
  "subtask": "left hand pick up plate rack from table ...",
  "next_step": "...",
  "video_key": "observation.images.head_stereo_left",
  "start_time": 0.0,
  "end_time": 10.0,
  "duration": 10.0,
  "timestamps": {"start": "00:00", "end": "00:10"},
  "source_chunk_video": "/mnt/raid0/supeng_g1_data/410_g1_ir/clear_plates/videos/observation.images.head_stereo_left/chunk-000/file-001.mp4",
  "source_from_timestamp": 217.26666666666668,
  "source_to_timestamp": 256.6,
  "source_timestamps": {"start": "03:37.27", "end": "04:16.60"},
  "video_2hz": "/mnt/raid0/.../episode_000017.mp4",
  "model": "/mnt/data/users/wangshuqi/models/Qwen3.6-27B-sft-merged",
  "status": "ok"
}
```

## Field semantics (confirmed)

| field | meaning |
|---|---|
| `goal` | the episode's natural-language instruction (one per task in practice) |
| `episode_id` / `episode_index` | episode identity within the task dataset |
| `step_index` | subtask position within the episode (0-based) |
| `subtask` | normalized subtask description (`subtask_raw_before_normalize` holds the model's original sentence) |
| `start_time` / `end_time` | **seconds relative to episode start**, whole-second resolution (they round-trip through the MM:SS `timestamps`) |
| `video_key` | camera stream — always `observation.images.head_stereo_left` in this file |
| `source_chunk_video` | server-absolute path to the concatenated chunk mp4 holding this episode, for `video_key` only |
| `source_from_timestamp` / `source_to_timestamp` | where this episode's footage begins/ends inside that chunk mp4 (seconds) |
| `status` | `"ok"` on every row observed; treated as a validity gate anyway |

**The deterministic segment mapping** (verified against actual footage):

```
chunk_video_time = source_from_timestamp + episode_time
```

so a subtask's footage is `source_chunk_video` from
`source_from_timestamp + start_time` to `source_from_timestamp + end_time`,
after translating the `/mnt/raid0/...` prefix to the local mount.
The pre-review JSONL is therefore **sufficient on its own** to resolve every
subtask to its video segment — no parquet lookups needed for the head camera.

**Camera trap (confirmed, do not forget when adding cameras):** LeRobot v3 splits
chunk files by *size*, so the same episode lands at a different `file-NNN.mp4`
and offset per camera. `source_chunk_video`/`source_from_timestamp` are valid
*only* for the row's own `video_key`. Other cameras must be resolved through
`meta/episodes/*.parquet` (see `~/Dev/evaluation/code/video_index.py` for a
working implementation). This project currently uses only the row's own camera.

## Validation results (all 34 tasks, 2026-08-11)

- 16,774 rows; every row `status == "ok"`; identical key sets within each file.
- 84 rows (0.5%) rejected by our loader: degenerate spans (`end <= start`) or
  `end_time` overshooting the episode's footage window by more than 1 s
  (MM:SS quantization makes ≤1 s overshoot legitimate; more is suspect).
- No missing chunk videos among sampled rows.
- One `goal` sentence per task (the 5-paraphrase variants live elsewhere and are not used).

## Confirmed / assumed / unresolved

**Confirmed**
- `open_vocab_subtasks.jsonl` is machine-generated, pre-human-review output.
- Timestamps are episode-relative seconds; the chunk-time mapping above was
  verified visually against extracted frames.
- Subtask boundaries are gap-free and contiguous by construction of the
  subtasker prompt (idle time belongs to the preceding subtask — a convention,
  not an observation).

**Assumptions**
- `status` values other than `"ok"` would mark unusable rows (none observed).
- 1 s tolerance on `end_time` overshoot is the right cutoff for MM:SS rounding.

**Unresolved**
- Whether wrist cameras will be needed for the evaluator input (the team's 235B
  judge used 3 views; the subtasker used head-only). Adding them requires the
  parquet-based camera resolution above.
- Boundary times are quantized to 1 s over a 2 Hz source — the effective
  precision floor for any boundary-quality evaluation.
- The pre-review annotations are *unvalidated model output* (the team's own
  accuracy summaries put time-IoU vs human review at 0.83–0.98 per task). They
  are inputs to be judged, never ground truth.
