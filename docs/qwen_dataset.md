# Qwen fine-tuning dataset export

How one subtask example becomes one record in the **official Qwen fine-tuning
format**, and what was verified about that format (2026-08-12, against
`QwenLM/Qwen3-VL` at `main`).

```
task JSONL ──▶ episodes ──▶ subtasks ──▶ EvaluationInput ──▶ Qwen record
(source.py)  (data/episodes.py)  (data/prepare.py)  (eval_input.py)  (qwen_export.py)
```

## The official format (verified facts)

The official framework is `qwen-vl-finetune/` inside the
[QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)
repo — the top-level README points to it, mentions no third-party trainer, and
ships a per-size script for our target model
(`scripts/sft_qwen3_4b.sh` → `Qwen/Qwen3-VL-4B-Instruct`). Its README and
`qwenvl/data/data_processor.py` are **byte-identical to Qwen2.5-VL's** (diffed):
the format did not change for Qwen3-VL.

One record, multi-image form (ours — one image per contact-sheet grid):

```json
{
  "image": ["ex_dir/grids/a_12f_tile336_1fps/grid_00.jpg", "…/grid_01.jpg"],
  "conversations": [
    {"from": "human", "value": "<image>\n<image>\n…question…"},
    {"from": "gpt",   "value": "…answer…"}
  ]
}
```

Rules enforced by the trainer's code (`data_processor.py::_build_messages`),
which `qwen_export.QwenSample` therefore validates at export time:

- `image` may be a string or list; each `<image>` tag in a human turn consumes
  the next image **left-to-right in list order**. Tag count ≠ image count is a
  hard error. Tags must not appear in the `gpt` turn.
- Roles are `from: human|gpt` only. **There is no system role** — a
  `{"from": "system"}` turn silently becomes an assistant turn and gets loss
  applied. The evaluation instruction must live inside the human turn.
- Only assistant spans are unmasked for loss; multi-turn is supported.
- Annotation file: `.json` (list of records) or `.jsonl` — both loaded, chosen
  by extension.
- Video (`"video"` + `<video>`) exists but takes **mp4 paths only**; a list is
  multiple videos, never frames-of-one-video, and there is **no per-sample
  fps/frame/timestamp field** — frame sampling is global CLI args
  (`video_fps`, `video_max_frames`, …). The framework gives the model no
  timestamp channel at all, which is exactly why our burned-in badge clock
  exists. We deliberately export **multi-image grids, not video**.
- Datasets are registered by editing a hard-coded dict in
  `qwen-vl-finetune/qwenvl/data/__init__.py`:
  `{"annotation_path": …, "data_path": …}`; media paths resolve as
  `data_path / <record path>` (`data_path` may be `""` for absolute paths).
  Selected at launch with `--dataset_use <name>`.
- Image resolution budget is **global**, not per-sample: `--max_pixels`
  (default `576*28*28`; the 4B demo script uses an aggressive `50176`). If
  `--max_pixels` is below a grid's raw pixel area the trainer downscales it —
  our baseline 12-frame grid is 1344×756 ≈ **1.02 MP**, so training needs
  `--max_pixels ≥ 1016064`. The exporter records the requirement as
  `manifest.max_grid_pixels` and the script prints it.

(Secondary reference: ms-swift instead uses `messages`/`images` keys and does
support a system role — different format, not what we target.)

## LLaMA-Factory: same records, different registration (verified facts)

The preferred trainer is **LLaMA-Factory**, and the export needs **no format
change** for it: LLaMA-Factory's ShareGPT defaults (verified 2026-08-12 in
`hiyouga/LLaMA-Factory` `src/llamafactory/data/parser.py::DatasetAttr`) are
exactly our record keys — `messages` column defaults to `"conversations"`,
tags default to `from`/`value` with `human`/`gpt`. Only the media column name
must be remapped. Registration = merge into `data/dataset_info.json`
(the export script prints this):

```json
"<name>": {
  "file_name": "<abs path>/annotations.json",
  "formatting": "sharegpt",
  "columns": {"messages": "conversations", "images": "image"}
}
```

Facts that matter, all from the LLaMA-Factory sources:

- **`<image>` count rule is identical**: placeholders summed over all messages
  must equal the images list length (hard `ValueError` in
  `mm_plugin.BasePlugin._validate_messages`), consumed left-to-right in list
  order. Our export already validates this shape.
- **Media path resolution**: relative paths are joined with `media_dir` — a
  **training-YAML/CLI argument** (`data_args.py`), NOT a `dataset_info.json`
  key; it defaults to `dataset_dir` (default `data/`). So the training config
  must set `media_dir: <export>/media`. Absolute `file_name` works.
- **Pixel budget**: `image_max_pixels` defaults to `768*768 = 589,824` and
  LLaMA-Factory resizes any larger image *before* the HF processor — our
  baseline 1344×756 grid (1,016,064 px) **is downscaled by default**; set
  `image_max_pixels: ≥ manifest.max_grid_pixels`. (The shipped
  `examples/train_lora/qwen3vl_lora_sft.yaml` even uses 512² — always override
  upward deliberately. The model-side processor allows far more; the LF knob
  is the binding one.)
- **Templates**: `qwen3_vl` (reasoning) / `qwen3_vl_nothink`; the official
  Qwen3-VL-4B-Instruct LoRA example uses `qwen3_vl_nothink`. Neither injects a
  default system prompt. Requires LLaMA-Factory **≥ v0.9.4** (Qwen3-VL support
  merged 2025-09-26).
- **System prompts ARE supported** (unlike qwen-vl-finetune): via a `system`
  column or a leading `{"from": "system"}` turn. We deliberately don't emit
  one, keeping the records valid for both trainers; if the rubric later wants
  a system prompt, LLaMA-Factory can take it without a format fork.
- **Silent-drop hazard**: samples with odd message counts or user turns at odd
  indices are *skipped with a log line*, not fatal. Our records are always
  strict human/gpt pairs (validated at export), so nothing can be dropped.
- **Keep `image` a list always** (we do): records mixing bare-string and list
  media would break Arrow schema inference at load time.
- Annotation may be `.json` (list) or `.jsonl`; a `file_name` may even be a
  directory of files. Video: LF additionally accepts pre-extracted frame lists
  as one "video" (nested list under `videos`) — but with no per-sample fps or
  timestamp field there either, frames get a synthesized clock from the global
  `video_fps`; our badge-clock grids remain the deliberate choice.

## Our export (`src/subtask_checker/qwen_export.py`)

`export_qwen_dataset(examples, out_dir, name, targets=None, …)` renders each
example's grids via the existing `eval_input.prepare_evaluation_input` (no
second grid algorithm) and writes a self-contained directory:

```
data/qwen_datasets/<name>/
├── annotations.json    official format; image paths relative to media/
├── media/<example>/
│   ├── grids/<config>/grid_*.jpg
│   └── evaluation_input.json      the inspectable intermediate
└── manifest.json       ours: resolved GridConfig/scope/instruction,
                        per-record provenance (annotation_source,
                        target_source, est tokens), skip lists, max_grid_pixels
```

The record's human turn is `HUMAN_TURN_TEMPLATE`: image tags (one per grid, in
grid order), the evaluation instruction, then the episode goal, the annotated
subtask description, and the claimed boundaries **on the burned-in badge
clock** — the only clock the model can see.

### Supervision targets

No supervision exists at this stage (hard rule: pre-review annotations are the
thing being judged, and no human-labelled data is read). Therefore:

- The assistant turn comes from an explicit `QwenTarget{text, source}`;
  `source` is recorded per record in the manifest, so supervision sources can
  never be silently mixed. `"ground_truth"` is a banned source name.
- Without targets, every record carries `PLACEHOLDER_TARGET`
  (`source="placeholder_pending_labels"`, loud `[NO TARGET YET]` text): the
  export is an input-side pipeline validation artifact, **not trainable**.
- `--targets file.json` (`{example_id: response_text}`) + `--target-source`
  fills real targets later; partial coverage is an error.

## Running it

```bash
uv run python scripts/export_qwen_dataset.py --name dev20                  # dev sample
uv run python scripts/export_qwen_dataset.py --name cp2 --tasks clear_plates --episodes 2
```

Selection is the prepared sample JSONL (default: dev sample, `--count` to
truncate) or a fresh task → episode → subtask traversal (`--tasks`, first
`--episodes` per task, via `data/episodes.py`). Exports above `--max-examples`
(default 50) are refused — raising the cap is a deliberate act, per the
small-sample rule. The script ends by printing both registration snippets
(LLaMA-Factory `dataset_info.json` entry + required YAML keys, and the
`qwenvl/data/__init__.py` dict entry).

`read_annotations()` / `missing_media_files()` re-validate an export from disk
— the same inspect-before-use path the rest of the pipeline follows.

## Episode layer (`src/subtask_checker/data/episodes.py`)

The source JSONL is flat; `EpisodeSubtasks` supplies the real hierarchy: one
episode = one goal, one video window, ordered unique subtask steps. Those
invariants were verified on the mount (clear_plates, collect_eggs,
pick_out_battery, zip_up: 0 violations) but are **validated, not assumed** —
`group_by_episode` raises on drift; `load_task_episodes` counts and reports
skipped rows/episodes like `build_sample` does.
