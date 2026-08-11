# Investigation report — data, subtasker, evaluation work, and the road to a Qwen3-VL-4B evaluator

*Written 2026-08-11. All statements are tagged **[confirmed]** (read from code/data), **[inferred]** (strongly implied), or **[unknown]** (needs verification). Paths verified on this machine.*

---

## 1. Executive summary

**What exists today.** The team runs an **annotation factory** (`Unitree_Robo_Describer`, sshfs-mounted from the GPU server `unitree@10.0.8.204`) over a **LeRobot v3.0** dataset of Unitree G1 humanoid demonstrations (`410_g1_ir`, ~33 manipulation tasks × ~1,000 episodes each, 30 fps, 8 camera streams). For a **~100-episode subset per task** (3,514 episodes total), a fine-tuned VLM (Qwen3.6-27B-sft, fed a whole 2 Hz / 320×240 episode video) segments each episode into subtasks with MM:SS boundaries and open-vocabulary descriptions. Humans review/correct those in a web viewer; the corrected file is authoritative and feeds two augmentation branches (21 instruction paraphrases per subtask via Qwen3.6-27B; one-sentence motion descriptions via Qwen3-VL-235B from 4 frames per subtask). The deliverable is `annotation_by_episode.json` per task.

**Your previous evaluation work** (`~/Dev/evaluation`) benchmarked and tried to improve the **LLM-as-judge** (Qwen3-VL-235B) that scores each generated subtask annotation against the video, using prompt iteration and GEPA prompt evolution against calibration labels *derived from* the human review diffs. Headline outcome after ~22 prompt versions and 5 GEPA rounds: prompt engineering hit a ceiling — the best honest configuration does not rank defective episodes better than a character count, GEPA winners were contaminated by task-specific memorisation, and the calibration labels themselves were shown to be ~⅓ noise. The repo's own conclusion: **"label archaeology, not prompt work"** — i.e. the missing ingredient is trustworthy evaluation labels, which is exactly what a fine-tuned evaluator needs too.

**What this means for the Qwen3-VL-4B evaluator.** All *inputs* of the desired training example already exist for ~3,510 reviewed episodes (15,808 reviewed subtask segments) across 33 tasks: videos, per-episode instruction (`goal`), reviewed segment boundaries + descriptions, and two battle-tested frame-grid implementations (the evaluation repo's `refiner_sheets` contact sheets with burned-in timestamps are the closest match to your target input). What does **not** exist is the *output* side: there is no rubric-style `{criteria, score, evidence, feedback}` dataset. What exists instead: 73 real human verdicts, ~505 Claude-generated verdicts for stack_cups, ~5,500 review-diff-derived good/edited labels (noisy), and 160 synthetic corruptions. Building the target dataset is therefore primarily a **label-construction problem**, not a modelling problem — and the review-diff data plus synthetic corruption machinery are the most promising raw material.

A working local training stack already exists: LLaMA-Factory + Qwen3-VL-4B-Instruct LoRA on the local 4090 was set up and verified (through model download) in `~/Dev/labelling_work/Subtask_Describer` for the motion-description task — the same harness can train the evaluator. **[confirmed]**

---

## 2. Dataset structure — `/home/xichao/remote/410_g1_ir`

### 2.1 What it is

- **[confirmed]** sshfs mount of `unitree@10.0.8.204:/mnt/raid0/supeng_g1_data/410_g1_ir` (checked via `mount`). This is why JSON files inside reference `/mnt/raid0/...` paths. The evaluation repo uses a *second* mount of the same data at `~/Dev/remote/410_g1_ir`.
- **[confirmed]** Format: **LeRobot `codebase_version: "v3.0"`**, `robot_type: unitree_g1` (per-task `meta/info.json`). Not v2.x: v3 concatenates many episodes into chunked files (`data/chunk-000/file-000.parquet`, `videos/<video_key>/chunk-000/file-XXX.mp4`) and `meta/episodes/*.parquet` records where each episode lives inside each chunk (per-camera `chunk_index`, `file_index`, `from_timestamp`, `to_timestamp`).
- Two side experiments sit next to the task dirs **[confirmed]**: `Motor_STAGE_0_1/` (raw zarr teleop captures, `ModeEnum.HUMAN/episode_*.zarr`) and two 2-episode `*_lerobot_v21_*smoke` dirs (LeRobot **v2.1** conversion tests, different cameras: `cam_left_wrist`…). These are unrelated to the subtask work. `TEST-clear_plates` is a test copy of a task dir.

### 2.2 Per-task layout (example: `clear_plates`)

```
clear_plates/
  data/chunk-000/file-000.parquet          # 30 Hz robot state/action table (~500 cols incl. per-frame timestamp)
  meta/
    info.json                              # v3.0, fps=30, features, camera calibration (intrinsics/extrinsics!)
    tasks.parquet                          # task_index -> instruction sentence
    episodes/chunk-000/file-000.parquet    # per-episode: tasks, length, per-camera chunk/file/from_ts/to_ts
    downsample.json                        # the 100 selected episode indices
    memory/episode_XXXXXX.json             # per-episode goal + subtask steps + rolling "memory" context (2 fps timeline)
    motion_desc_pipeline/                  # ALL subtasker outputs (see §3)
  videos/<video_key>/chunk-000/file-XXX.mp4   # AV1 640x480 30fps, episodes concatenated (~8.5 min per file)
```

### 2.3 Cameras **[confirmed]**

8 video streams per task, all 640×480 AV1 30 fps: `head_stereo_left`, `head_stereo_right`, `wrist_left`, `wrist_right`, `wrist_left_ir`, `wrist_right_ir`, `head_stereo_left_rec`, `head_stereo_right_rec` (rectified). Full stereo + wrist camera **calibration matrices are in `info.json`**. The subtasker uses only `head_stereo_left`; the 235B judge uses head + both wrists.

**Camera-mapping trap [confirmed, hard-won]:** chunk files are split by *size*, not time — the same episode lands in *different* `file_index`/`from_timestamp` per camera. Naively swapping the camera directory in a path reads footage from a different part of the session. The evaluation repo fixed this on 2026-08-07 (`code/video_index.py` resolves each camera through `meta/episodes/*.parquet`); every judge number before that fix was measured on partly-wrong wrist footage.

### 2.4 Instructions, timestamps, splits

- **Instruction**: every episode carries a natural-language instruction in `episodes.parquet:tasks` / `meta/tasks.parquet` (e.g. *"Place the plates on the rack from right to left in the order of red, green, and blue."*). The 2 Hz manifest additionally carries `high_level_task_raw` with 5 `@`-separated paraphrases. **[confirmed]**
- **Timestamps**: per-frame `timestamp` column in data parquet; per-episode per-camera `from/to_timestamp` into chunk videos; subtask boundaries in seconds relative to episode start (see §3). **[confirmed]**
- **Splits**: `info.json` declares only `train: 0:N` for every task. **There are no train/val/test splits anywhere in the dataset.** The only split ever constructed is in the evaluation repo (calibration train/heldout, stratified by task+label, seed 42) and in `Subtask_Describer/training` (466/25 by episode). **[confirmed]**

### 2.5 Statistics **[confirmed — collected by `scripts/dataset_stats.py`, full JSON in `extracted/dataset_stats.json`]**

Sweep of every task dir's `meta/` (info.json, downsample.json, memory/*.json, accuracy summaries):

```
Task datasets (LeRobot v3.0):        33   (+ TEST-clear_plates copy, + 2 tiny v2.1 smoke dirs, + raw zarr Motor_STAGE_0_1)
Total episodes:                  33,653   (typically ~1,000/task; color_arrangement 1,802; collect_eggs 199)
Total footage (head cam):        ~169 h   (×8 camera streams recorded; 30 fps, 640×480 AV1)
Episodes selected for annotation: 3,514   (downsample.json; 100/task typical — water_plant 201, collect_eggs_0508 148, collect_eggs 53)
memory/ episode JSONs:            3,510
Reviewed subtask segments:       15,808
Subtasks per episode:              4.57 mean   (task means 2.29 – 6.75)
Subtask duration:                  4.14 s mean, 4.0 s median, min 0.0 s (!), max 44.0 s
Frame rate / resolution:           30 fps 640×480 (source); 2 Hz 320×240 (subtasker input)
Train/val/test split:              none declared (info.json: train 0:N only)
```

Notes: minimum duration 0.0 s means degenerate zero-length segments exist in the reviewed data (filter before training). Where the team's accuracy summary exists, model-vs-review time-IoU is 0.83–0.98 (clear_plates 0.873, collect_eggs 0.825, color_arrangement 0.983, disconnect_ethernet 0.884). `collect_eggs_0508` and `toolbox_storage_hard` are variant recordings of their base tasks. Full per-task table: `extracted/dataset_stats.json`.

| task | eps | hours | reviewed eps | segments | steps/ep | med dur (s) |
|---|---|---|---|---|---|---|
| clear_plates | 1034 | 6.1 | 100 | 491 | 4.9 | 4.0 |
| collect_eggs_0508 | 801 | 3.5 | 148 | 794 | 5.4 | 3.0 |
| color_arrangement | 1802 | 7.2 | 100 | 315 | 3.2 | 4.0 |
| fold_towel | 1015 | 7.4 | 100 | 493 | 4.9 | 5.0 |
| pack_iphone | 995 | 8.2 | 100 | 594 | 5.9 | 5.0 |
| spell_Unitree | 1005 | 8.8 | 100 | 401 | 4.0 | 8.0 |
| store_racket | 999 | 9.7 | 100 | 504 | 5.0 | 6.0 |
| water_plant | 1006 | 4.9 | 201 | 678 | 3.4 | 5.0 |
| … 25 more tasks (see extracted JSON) | | | | | | |

### 2.6 `meta/memory/` — a third artifact family

Per-episode JSONs (one per downsampled episode) with the reviewed subtask sequence *plus* a rolling context field: `{step_index, subtask, goal, memory_des, start_frame, end_frame, start_time, end_time, video_fps: 2.0}`. Boundaries match the reviewed subtask file exactly; frame indices are in the **2 Hz** episode-video timeline. **[confirmed]** Owned by a different server user and produced by code that is *not* in `Unitree_Robo_Describer` — plausibly the high-level-VLA training-data builder consuming the reviewed subtasks. **[inferred; provenance unknown]**

---

## 3. The subtasker — `Unitree_Robo_Describer` (read-only team repo)

sshfs mount of `unitree@10.0.8.204:/mnt/raid0/motion_data/motion_des_wsq/Unitree_Robo_Describer`; a fast local snapshot lives at `~/Dev/labelling_work/Unitree_Robo_Describer_local` (incl. `ONBOARDING.md`, written by you and verified against the server). Branch `feature/subtask-eval-qwen235b-true3view`, 11 commits, heavy uncommitted changes and `.bak_*` files (the team versions by timestamped copies, not commits). **[confirmed]**

### 3.1 Pipeline (driver: `scripts/run_full_lerobot_v3_realtime.sh`, env-var gated)

```
LeRobot v3 task dir  (videos/<head_stereo_left>/chunk-*/file-*.mp4 + meta/episodes parquet)
   │  STEP 00–02b  ffmpeg: cut per-episode clips  → 2 Hz, 0.5× scale (320×240 h264)
   │               [DOWNSAMPLE_JSON = meta/downsample.json → only the 100 selected episodes]
   ▼
   STEP 03   Qwen3.6-27B-sft-merged (vLLM :9020), temperature 0, whole 2 Hz episode video
   │         base64-inlined as video_url; prompt scripts/subtask_prompt/open_vocab_prompt_full.txt
   │         → {"subtasks":[{"name", "timestamps":{"start":"MM:SS","end":"MM:SS"}}]}
   │         gap-free, boundary-contiguous, atomic verb_object naming; ≤10 parse retries
   ▼
   STEP 03b/bb/bbb  import (MM:SS→sec) → rule normalization (rule_canonical_v2)
   │         → open_vocab_subtasks.jsonl (pre-review)
   │         → open_vocab_subtasks_results_normalized_for_review.json  ← HUMAN REVIEW (web viewer,
   │           writes back in place; .bak_before_websave)   ★ authoritative from here on:
   │           choose_subtask_source() prefers the review JSON everywhere downstream
   ├─ BRANCH A  Qwen3.6-27B (:9010): 21 augmented instructions per unique subtask
   │            (11 noise types incl. 2 deliberate negatives; 4 frames attached; reuse cache)
   └─ BRANCH B  ffmpeg: 1 Hz per-subtask clips (segments_1hz) → 4 evenly-spaced PNGs (frames_4)
                → Qwen3-VL-235B-A22B-Instruct (:9030): one-sentence motion description
   ▼
   meta_writer join on (episode_id, step_index)
   → annotations.json (flat) → annotation_by_episode.json (deliverable)
```

- The two log families in the dataset dir map to this: `<task>_subtask_downsample.log` = STEP 00–03bbb over the 100-episode subset; `<task>_noise_motion_235b_review_json_*.log` = rerun with `RUN_SUBTASK=0` after human review, regenerating noise+motion from the reviewed subtasks. **[confirmed from log CONFIG banners]**
- **No training code exists in the repo** — pure inference against three local vLLM endpoints. The subtask model is *named* `-sft-merged`, so someone fine-tuned it elsewhere. **[confirmed absence; provenance of the SFT unknown]**
- An experimental two-stage route exists (235B macro-segmentation at 0.5 fps → 27B-sft subtasks in 40 s windows with 10 s overlap → stitching/continuity repair), tested on a different dataset (`demodata_lerobotv3`), not used for `410_g1_ir`. **[confirmed]**

### 3.2 Timestamp bookkeeping (the part your evaluator must reproduce)

- Subtask boundaries are **seconds relative to episode start**, quantised to whole seconds by the MM:SS format (nearest-second labels; 2 Hz source → 0.5 s native resolution).
- Mapping to the full-res footage: `chunk_video_time = source_from_timestamp + episode_time`, per camera via `meta/episodes/*.parquet`. **[confirmed]**

### 3.3 Key schemas

Per-subtask row (`open_vocab_subtasks.jsonl` / review JSON — the judge's input):

```json
{"dataset":"410_g1_ir","task":"clear_plates",
 "goal":"Place the plates on the rack from right to left in the order of red, green, and blue.",
 "episode_id":"episode_000017","episode_index":17,"step_index":0,
 "subtask":"left hand pick up plate rack from table ...","next_step":"...",
 "video_key":"observation.images.head_stereo_left",
 "start_time":0.0,"end_time":10.0,"duration":10.0,
 "timestamps":{"start":"00:00","end":"00:10"},
 "source_chunk_video":".../videos/observation.images.head_stereo_left/chunk-000/file-001.mp4",
 "source_from_timestamp":217.267,"source_to_timestamp":256.6,
 "video_2hz":".../_episode_video_cache/.../episode_000017.mp4",
 "model":".../Qwen3.6-27B-sft-merged","status":"ok",
 "subtask_raw_before_normalize":"Left hand picks up the plate rack on the table. ...",
 "original_subtask":"...","subtask_normalized":"...",
 "subtask_normalization_key":"...","subtask_normalization_method":"rule_canonical_v2"}
```

Deliverable (`annotation_by_episode.json`): per episode `{episode_id, episode_index, high_level_task, subtask:[{task, timestamps{MM:SS}, duration, motion, augmented_subtasks:[{type, content}×21]}]}`. Note `high_level_task` degrades to the folder name ("clear plates") — the rich goal sentence is dropped at export. **[confirmed]**

Team-side quality numbers (model pre-review vs human post-review, `subtask_accuracy_summary.json`, clear_plates): time-IoU **0.873**, LLM content similarity **0.80**, step-count match **0.87**, exact-string 0.0. So human review changes something in most episodes but boundaries are mostly close. **[confirmed]**

---

## 4. Previous evaluation work — `~/Dev/evaluation`

An LLM-as-judge benchmarking workspace (Aug 4–10), 45 commits. The objective evolved: **v1–v3** precision-tuned defect flagger (metric F0.5) → **v4** recall-first filter/ranker (`ranking_quality`, `depth_to_recall_95`) → **v5** certification ("would I stake the dataset on this segment being correct" — certify vs send-to-review, target purity 95%). **[confirmed]**

### 4.1 Mechanics

- Judge = **Qwen3-VL-235B** on `:9030` via SSH tunnel (no API keys; plain urllib against the OpenAI-compatible vLLM endpoint), temperature 0, max_tokens 2048, concurrency 8.
- Input per segment: system prompt (one of 22+ versions in `prompts/`) + user prompt built in `judge_core.py` (goal, task-statistics card, candidate `{name, timestamps}`, camera order) + frames (two modes, §5).
- Output: strict JSON. Seed schema: overall `score` 1–10 + six weighted dimension scores (visual 30%, semantic 20%, temporal 20%, body_part 15%, granularity 10%, goal 5%) + `issues_found` + `suggested_human_label`. v5 adds observation fields (`action_start_observed_sec`, `actions_observed[]`, `text_defects{}`, `certification{verdict, checked[], could_not_check[]}`) with Python, not the model, computing boundary error and severity (weights: boundary .30, granularity .25, text .20, hand .15, actions .07, goal .03). **[confirmed]** — this "measure, don't opine" schema is the closest existing ancestor of your `{criteria, score, evidence, feedback}` target.
- GEPA: standalone `gepa==0.1.4` (not DSPy), evolving only the system prompt; reflection LM = the same 235B; asymmetric metric (miss 0.3, false alarm 0.0); train-split-only guardrails with a one-shot heldout tripwire.

### 4.2 Ground truth (the crux)

| set | n | source |
|---|---|---|
| `calibration_set.json` | 856 segments, 3 tasks | review-diff derived |
| `calibration_v2_multitask.json` | 1,516 segments, 14 tasks | review-diff derived |
| `labels_v2.json` | 5,565 segments, 12 tasks | review-diff derived + provenance filters |
| `episode_gt_v1.json` | 1,386 episodes (36.8% need edit) | review-diff derived |
| `viewer_verdicts.jsonl` | **73 real human verdicts** (46 bad/27 good) | human, via viewer |
| `wgo_bench` | 903 (743 gold + **160 synthetic corruptions**: hand_swap, description_swap, boundary_shift, merge_two ×40) | external + scripted |
| claude_labeller (stack_cups) | 505 segments, 101 episodes; keep/fixed/flagged verdicts with frame-cited evidence | Claude overnight review (partially human-audited) |

Label semantics are **"a reviewer changed this text," not "this is wrong"** — audits found only 66% of edited segments paired trustworthily and ~⅓ of the judge's apparent misses were the label being wrong, not the judge. **[confirmed from `build_labels_v2.py`, `build_gold_queue.py` docstrings]**

### 4.3 Results & lessons (measured, not opinions)

- Seed judge nearly blind by design (recall 4.8%, F0.5 0.14) — caused by an explicit "~3% prior" in the team prompt; removing that block alone quadrupled recall.
- Best honest hand prompt F0.5 0.39; best GEPA prompt 0.55 on trained tasks — but **contaminated** (per-task cheat sheets, memorised strings, label-provenance leakage); unseen-task recall 2.7% vs 32.4% trained. Generality checker + reflection constraints were added in response.
- **Headline negative result:** v4's per-dimension signals are all at chance (AUC ≈ 0.5); episode ranking_quality −0.105; **text length separates edited from kept segments better than the 235B judge**. Written conclusion: *"more prompt engineering is not the lever… next step should be label archaeology."*
- v5 certification: coverage 39.1% at purity 86.5% (target 95% — not met).
- Boundary-time fabrication was caught: the judge "read" boundary errors that were exactly ± the sampling interval; fixed by demanding times copied from burned-in frame badges and validating against frames actually sent (`OBSERVED_TIME_TOL_SEC=0.3`).
- Payload limits for the shared vLLM engine: 336 px tiles ×12 (≈3 MPx/request) fine; 504 px ×19 (≈13.7 MPx) **crashed the engine** — guards now cap 2 MB / 6 MPx per request. The final "richframes" run scored 0/145.
- The frame A/B experiment (grids vs loose frames, same prompt) was built but **never run** — representation choice remains unmeasured and confounded with prompt version. **[confirmed]**

---

## 5. Video-grid pipeline (in `~/Dev/evaluation`)

Two modes, switched by `FRAME_MODE` in `code/config.py` (default still `legacy_3view`):

| | Mode A `legacy_3view` | Mode B `refiner_sheets` (the grids) |
|---|---|---|
| Code | `judge_core.py` | `refiner_frames.py` + `macrodata-refiner` 0.3.7 |
| Scope | claimed span only | span **± 3 s context pad** |
| Sampling | 1 Hz, then uniform cap at **4 timestamps** | adaptive: window/(12−1) steps, min 0.5 s (matches 2 Hz source) |
| Form | ≤12 loose JPEGs (4 ts × 3 cameras), 640×480 native, `detail: low` | one **4-col contact sheet per camera** (3 sheets), 12-tile budget → 4×4 grid, 336 px tiles → **1344×1008 JPEG**, q95, `detail: high` |
| Timestamps | text next to each image only | **burned into pixels**: black badge top-left, `003.00s`, rebased to clip start |
| Long segments | coarser effective step | same 12 tiles stretched (coarser sample_sec) |
| Blind spot | actions starting before the span are invisible by construction | pad shows 3 s of context; >12-tile actions still undersampled |
| Cache | `results/frame_cache/<task>/…json` — 4,103 files, 2.3 GB, content-addressed (excludes prompt → prompt edits stay cache-hot; includes geometry + camera-fix marker) | same cache, `refiner_sheets_` prefix (1,899 sheet entries) |

Both extract frames with **ffmpeg** (`-ss`) from the original AV1 chunk videos at `source_from_timestamp + t` — resolved *per camera* through `meta/episodes/*.parquet` (`code/video_index.py`).

**Suitability for Qwen3-VL [assessment]:** the sheet representation is a good fit — Qwen3-VL handles tiled images with burned-in timestamps well, and the "copy the badge time" protocol gave the honest boundary measurements. Known losses: 0.5–1 s+ temporal resolution (grows with segment length), 336 px tiles (~⅛ the pixel area of source), black padding tiles, JPEG artifacts, and single-sheet truncation of long segments. The 4B model's smaller vision budget makes the 3-sheet (3-camera) input ≈ 4 MPx/example a real cost consideration; head-only (1 sheet) is the cheap variant but the team's own evaluator went 3-view because wrist cams catch grasp errors.

Upstream (server-side) representations that also exist: whole-episode 2 Hz 320×240 mp4s (subtasker input), 1 Hz per-subtask clips, 4-PNG-per-subtask sets (motion/noise input), and the judge's 12-loose-JPEG mode.

---

## 6. Current end-to-end data flow (all `?` filled)

```
RAW FOOTAGE   410_g1_ir/<task>/videos/<8 cameras>/chunk-*/file-*.mp4  (LeRobot v3, AV1 640×480@30)
     ↓   ffmpeg cut per episode via meta/episodes parquet (from_timestamp), head_stereo_left only,
     ↓   fps=2, scale=0.5  → 320×240 2 Hz per-episode mp4     [STEP 00–02b; DOWNSAMPLE_JSON → 100 eps/task]
SUBTASK GENERATION
     ↓   whole 2 Hz video base64 → Qwen3.6-27B-sft (:9020), open_vocab prompt, temp 0
     ↓   → {"subtasks":[{name, timestamps MM:SS}]}  → import → normalize (rule_canonical_v2)
SUBTASK JSON
     ↓   open_vocab_subtasks.jsonl  (pre-review)
     ↓   → HUMAN REVIEW in web viewer → open_vocab_subtasks_results_normalized_for_review.json  ★ authoritative
     ↓   → (team) noise ×21 + motion descriptions → annotations.json → annotation_by_episode.json
     ↓   → (team) meta/memory/episode_*.json  (reviewed steps + rolling memory context)
VIDEO GRID   (evaluation repo, per segment to be judged)
     ↓   resolve camera-correct chunk file + offset (video_index.py)
     ↓   ffmpeg extract at source_from_timestamp + t
     ↓   legacy: ≤12 loose JPEGs   |   sheets: 3 × (4×4 contact sheet, 336px tiles, badge times, ±3s pad)
     ↓   content-addressed cache (results/frame_cache/)
EVALUATION PROMPT
     ↓   system = prompts/judge_v*.md   +   user = goal, task-stats card, candidate {name, MM:SS}, camera info
     ↓   → Qwen3-VL-235B (:9030, temp 0, JSON forced)
EVALUATION OUTPUT
         v5 JSON: observed times/actions/defect booleans + certification verdict + 1–10 scores
     ↓   Python post-processing: boundary error, severity, episode roll-up, ranking metrics
         → calib_results.jsonl → metrics / review queue (worst-first for humans)
```

---

## 7. Proposed canonical training-example schema

Framework-independent; every field maps to an existing artifact except `evaluation`, which is the gap (§8). One example = one segment of one episode.

```json
{
  "example_id": "410_g1_ir/clear_plates/episode_000017/step_0003",
  "source": {
    "dataset": "410_g1_ir", "task": "clear_plates",
    "episode_id": "episode_000017", "episode_index": 17, "step_index": 3,
    "video_key": "observation.images.head_stereo_left",
    "chunk_video": "videos/observation.images.head_stereo_left/chunk-000/file-001.mp4",
    "chunk_offset_sec": 217.267,
    "episode_duration_sec": 39.33,
    "annotation_provenance": "model_pre_review | human_reviewed | synthetic_corruption"
  },
  "instruction": "Place the plates on the rack from right to left in the order of red, green, and blue.",
  "segment": { "start_sec": 18.0, "end_sec": 22.0,
               "description": "right hand place red plate on right side of plate rack" },
  "context": {
    "prior_steps": ["...step 0..2 descriptions..."],          // optional; memory/ files have this
    "episode_step_count": 7
  },
  "frames": {
    "mode": "contact_sheet",
    "images": ["sheets/head.jpg", "sheets/wrist_left.jpg", "sheets/wrist_right.jpg"],
    "grid": [4, 4], "tile_px": [336, 252],
    "sampling": "adaptive<=12 tiles over span±3s, min 0.5s",
    "timestamps_burned_in": true, "timestamp_origin": "segment_window_start",
    "badge_times_sec": [0.0, 1.0, "..."]
  },
  "evaluation": {                                             // ← the part that must be BUILT
    "criteria": [
      { "criterion": "boundary_accuracy",
        "score": 2, "max_score": 5,
        "evidence": "badge 07.00s: gripper already holds plate before claimed start; release visible at badge 03.00s of next window",
        "feedback": "Start ~2s late; move start to 16s." },
      { "criterion": "description_accuracy",  "score": 5, "evidence": "...", "feedback": "..." },
      { "criterion": "body_part_correctness", "score": 5, "evidence": "...", "feedback": "..." },
      { "criterion": "granularity",           "score": 5, "evidence": "...", "feedback": "..." },
      { "criterion": "goal_alignment",        "score": 5, "evidence": "...", "feedback": "..." }
    ],
    "overall_score": 3.4,
    "verdict": "REVIEW",                                      // CERTIFY | REVIEW — keep v5's actionable target
    "defect_tags": ["bad_boundary"],                          // team's existing review taxonomy
    "label_provenance": "human | review_diff | synthetic | model_consensus"
  }
}
```

Design notes grounded in the investigation:

- Keep **times in seconds relative to the segment window** and burn them into frames — the v5 work proved the model fabricates boundary errors otherwise.
- The five criteria mirror the team's review taxonomy (`bad_boundary / bad_description / bad_body_part / over- / under_segmented / goal`) and the v4 severity dimensions, so review-diff data can supervise them directly.
- Carry `annotation_provenance` and `label_provenance` on every example: the GEPA contamination incident shows provenance must never be inferable from content, and you'll want to weight human > derived > synthetic at training time.
- A `verdict` field (certify/review) is more actionable than a bare score and matches the deployment need (triage of the review queue).

## 8. Missing data

```
✓ Videos (8 cameras, calibrated, 30 fps; per-camera episode offsets)
✓ Overall instruction per episode (+5 paraphrases; 21 per-subtask variants)
✓ Segment boundaries (reviewed, second-resolution): 15,808 segments, ~3,510 eps, 33 tasks
✓ Subtask descriptions (raw + normalized + human-reviewed)
✓ Motion descriptions (235B, one sentence per segment)
✓ Frame-grid generator + 2.3 GB cache (but cache keyed to old geometry decisions)
✓ Pre-review vs post-review diffs (weak defect labels, ~5.5k segments)
✓ 160 synthetic corruptions (4 types) + generator pattern (wgo_bench)
△ 73 human verdicts + 505 Claude verdicts (stack_cups) + 1 fully-corrected task (plug_in_ethernet)
✗ Rubric-style criteria/score/evidence/feedback labels        ← must be built
✗ Trustworthy per-segment "is it actually wrong" labels at scale (review-diff ≈ 66% trustworthy)
✗ Calibrated overall scores (human 1–10 or graded verdicts)
✗ Evidence annotations tied to specific frames/times (exists only in 73+505 verdicts)
✗ Ground-truth boundary quality finer than 1 s (source is 2 Hz + MM:SS rounding)
✗ Declared train/val/test split (must be created, by task AND by episode)
✗ Failed-attempt/retry annotations (one commit says "retries count as ONE action" — policy, not labels)
```

## 9. Recommended next steps

1. **Freeze an evaluation split first.** Hold out entire *tasks* (e.g. 3–4 of the 29) plus held-out episodes within training tasks, before generating any labels. The GEPA leakage incident is the cautionary tale.
2. **Build the label set from what review already paid for** (label archaeology, as your own notes concluded):
   - Re-derive review-diff labels with the `labels_v2` provenance filters (drop the ~34% untrustworthy pairings); map each edit type to the criteria fields (text edit → description_accuracy; boundary move → boundary_accuracy; split/merge → granularity).
   - Promote the 73 viewer verdicts + audited claude_labeller verdicts to gold; their frame-cited `reason` text is exactly the `evidence`/`feedback` you need.
   - **Scale synthetic negatives**: extend wgo_bench's 4 corruption generators (hand_swap, description_swap, boundary_shift ±k s, merge/split) over the ~15.8k reviewed segments (the review-surviving majority are "good" anchors). Corruptions give *known* defect type, magnitude, and clean evidence ("the description says left hand; frames show right") — ideal for teaching criteria/evidence, and label-noise-free by construction.
   - Optionally distil graded feedback text with the 235B *given the answer* (tell it what the defect is; ask only for evidence/feedback prose citing badge times) — generation-conditioned-on-truth avoids the judge's known blindness.
3. **Fix the representation before mass-generating grids:** run the already-built frame A/B (`build_frame_ab_queue.py`) sheets-vs-loose on the 73+505 gold verdicts with the 235B, pick one mode, and stay under the proven-safe geometry (336 px, ≤12 tiles). Consider one head sheet + wrists only for hand-sensitive criteria to cut tokens for the 4B model.
4. **Reuse the existing local training stack** (`Subtask_Describer/training`): LLaMA-Factory LoRA, `qwen3_vl` template, images list per sample — the evaluator dataset drops in as `{messages, images}` with the JSON evaluation as the assistant turn. Smoke-run first (Stage 4 of that project's plan, still pending).
5. **Validate the student against the only trustworthy references**: held-out human verdicts and held-out synthetic corruption types (train on 3 corruption generators, test on the 4th to measure generalisation, not memorisation of corruption fingerprints).

## 10. Risks and open questions

- **Reviewed ≠ ground truth.** Human review was itself triaged by the lenient seed judge and known to over-flag ~25% in the team's own summary; treated-as-good segments include unexamined ones. Certify-grade labels need the provenance filtering of step 2.
- **Boundary ambiguity is intrinsic**: 1 s label quantisation over 2 Hz source; the reviewed data enforces gap-free contiguous timelines, so "boundary correct" is partly a convention (idle time belongs to the *preceding* subtask) the evaluator must be taught, not discover.
- **~45% of real defects are purely temporal** (boundary/segmentation) — the hardest thing to see in a 12-tile grid; long segments dilute sampling further. Temporal criteria may need denser tiles or per-criterion frame windows.
- **Generated subtasks are NOT safe to treat as ground truth** — 36.8% of episodes needed edits; pre-review rows are the *thing being judged*, never the reference.
- **Leakage surfaces everywhere**: task-specific phrasing memorisation (GEPA precedent), synthetic-corruption fingerprints, and identical instructions across ~1,000 episodes of a task make by-episode splits weak; by-task holdout is the honest test.
- **Contiguity conventions vs failed attempts**: retries are annotated as one action by policy; an evaluator judging "did the action succeed" will meet segments containing failed attempts labelled as single successes.
- **Engine/token budget**: 3-camera sheets ≈ 1 MB/3 sheets/example at 336 px; the 504 px experiment crashed the shared engine and parsed 0/145. For a 4B student, start at 336 px and measure.
- **Open questions:** Who produces `meta/memory/` and is it maintained? Is the macro-window segmentation route going to replace single-pass (changing the distribution the evaluator sees)? Are wrist cameras required for acceptable hand-attribution accuracy at 4B scale? How many human gold labels are enough to calibrate scores — 73 is clearly too few for per-criterion calibration?

---

*Companion artifacts in this directory: `scripts/dataset_stats.py` (sweep script), `extracted/dataset_stats.json` (full per-task statistics).*
