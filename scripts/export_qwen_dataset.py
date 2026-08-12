#!/usr/bin/env python3
"""Export subtask examples as an official-format Qwen fine-tuning dataset.

    uv run python scripts/export_qwen_dataset.py --name dev20                  # from the dev sample
    uv run python scripts/export_qwen_dataset.py --name cp2 --tasks clear_plates --episodes 2
    uv run python scripts/export_qwen_dataset.py --name dev20 --grid-config b_25f_tile256_2fps

Selection is either the prepared sample JSONL (default: the dev sample) or a
fresh task → episode → subtask traversal of the mount (--tasks, first
--episodes episodes per task). Targets: without --targets every record carries
the loud PLACEHOLDER_TARGET (input-side export, not trainable); --targets is a
JSON object {example_id: response_text} and requires --target-source naming
the supervision source.

Output: <out-root>/<name>/ with annotations.json + media/ + manifest.json
(docs/qwen_dataset.md). The records are valid for BOTH trainers — LLaMA-Factory
(ShareGPT) and the official qwen-vl-finetune — and both registration snippets
are printed at the end.
"""

import argparse
import json
from pathlib import Path

from subtask_checker.config import PATHS
from subtask_checker.data.episodes import iter_task_episodes
from subtask_checker.data.prepare import read_sample_jsonl
from subtask_checker.eval_input import DEFAULT_EVALUATION_INSTRUCTION, DEFAULT_GRID_CONFIG
from subtask_checker.qwen_export import (
    QwenTarget,
    export_qwen_dataset,
    llama_factory_snippet,
    missing_media_files,
    registration_snippet,
)
from subtask_checker.video.gridset import CANDIDATE_CONFIGS

DEFAULT_OUT_ROOT = PATHS.output_root / "qwen_datasets"
# Hard rule 5: small samples while the pipeline is under development. Raising
# this is a deliberate act, not a default.
DEFAULT_MAX_EXAMPLES = 50


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", required=True, help="dataset name (directory + registry key)")
    src = ap.add_argument_group("selection")
    src.add_argument("--sample", type=Path, default=PATHS.dev_sample_jsonl,
                     help="sample JSONL of TaskExamples (default: dev sample)")
    src.add_argument("--count", type=int, default=None,
                     help="use only the first N examples of the sample")
    src.add_argument("--tasks", nargs="+", default=None,
                     help="instead of --sample: traverse these tasks on the mount")
    src.add_argument("--episodes", type=int, default=1,
                     help="with --tasks: episodes per task (default 1)")
    ap.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES,
                    help=f"refuse larger exports (default {DEFAULT_MAX_EXAMPLES})")
    ap.add_argument("--grid-config", default=DEFAULT_GRID_CONFIG.name,
                    choices=[c.name for c in CANDIDATE_CONFIGS])
    ap.add_argument("--scope", choices=["segment", "episode"], default="segment")
    ap.add_argument("--instruction", default=DEFAULT_EVALUATION_INSTRUCTION)
    ap.add_argument("--targets", type=Path, default=None,
                    help="JSON object {example_id: response_text}")
    ap.add_argument("--target-source", default=None,
                    help="supervision source label, required with --targets")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = ap.parse_args()

    if args.tasks is not None:
        examples = []
        for task, episodes, report in iter_task_episodes(tasks=args.tasks):
            for ep in episodes[: args.episodes]:
                examples.extend(ep.subtasks)
            if report["invalid_rows"] or report["invalid_episodes"]:
                print(f"note: {task}: skipped {report['invalid_rows']} row(s), "
                      f"{len(report['invalid_episodes'])} episode(s) while loading")
    else:
        examples = read_sample_jsonl(args.sample)
        if args.count is not None:
            examples = examples[: args.count]
    if not examples:
        raise SystemExit("no examples selected")
    if len(examples) > args.max_examples:
        raise SystemExit(
            f"{len(examples)} examples exceed --max-examples {args.max_examples}; "
            "narrow the selection or raise the cap deliberately"
        )

    targets = None
    if args.targets is not None:
        if not args.target_source:
            raise SystemExit("--targets requires --target-source")
        raw = json.loads(args.targets.read_text(encoding="utf-8"))
        targets = {
            k: QwenTarget(text=v, source=args.target_source) for k, v in raw.items()
        }

    grid_config = next(c for c in CANDIDATE_CONFIGS if c.name == args.grid_config)
    out_dir = args.out_root / args.name
    manifest = export_qwen_dataset(
        examples, out_dir, name=args.name, targets=targets,
        grid_config=grid_config, scope=args.scope, instruction=args.instruction,
    )

    missing = missing_media_files(out_dir)
    if missing:
        raise SystemExit(f"referenced images missing after export: {missing[:5]}")

    print(f"dataset        {args.name}  ({manifest.scope} / {manifest.grid_config.name})")
    print(f"records        {len(manifest.records)} "
          f"(skipped: {len(manifest.skipped_missing_video)} missing video, "
          f"{len(manifest.skipped_render_failed)} render failures)")
    print(f"target source  {', '.join(manifest.target_sources)}")
    if manifest.target_sources == ["placeholder_pending_labels"]:
        print("               input-side export — NOT trainable until real targets exist")
    print(f"wrote          {out_dir}/{{annotations.json, manifest.json, media/}}")
    print(f"note           trainer pixel budget must be >= {manifest.max_grid_pixels} "
          "or the grids get downscaled\n"
          "               (LLaMA-Factory image_max_pixels, default 589824; "
          "qwen-vl-finetune --max_pixels)")
    print("\nregistration — LLaMA-Factory (merge into data/dataset_info.json; >= v0.9.4):\n")
    print(llama_factory_snippet(args.name, out_dir))
    print(f"\n  training yaml: template: qwen3_vl_nothink   "
          f"media_dir: {out_dir.resolve() / 'media'}   "
          f"image_max_pixels: {manifest.max_grid_pixels}")
    print("\nregistration — official qwen-vl-finetune (qwenvl/data/__init__.py):\n")
    print(registration_snippet(args.name, out_dir))


if __name__ == "__main__":
    main()
