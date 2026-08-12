#!/usr/bin/env python3
"""Prepare the complete INPUT of one model evaluation example — no model call.

    uv run python scripts/prepare_input.py --index 0
    uv run python scripts/prepare_input.py --example-id 410_g1_ir/clear_plates/episode_000617/step_005
    uv run python scripts/prepare_input.py --index 0 --grid-config b_25f_tile256_2fps --scope episode

Output under data/samples/examples/<example_id>/ (the same directory
prepare_example.py uses, so everything about one segment stays together):
    evaluation_input.json          the validated EvaluationInput
    grids/<config_name>/grid_*.jpg the grid images it references (relative paths)
"""

import argparse
from pathlib import Path

from subtask_checker.config import PATHS
from subtask_checker.data.prepare import read_sample_jsonl
from subtask_checker.eval_input import (
    DEFAULT_EVALUATION_INSTRUCTION,
    DEFAULT_GRID_CONFIG,
    missing_visual_files,
    prepare_evaluation_input,
    write_evaluation_input,
)
from subtask_checker.video.gridset import CANDIDATE_CONFIGS

DEFAULT_SAMPLE = PATHS.dev_sample_jsonl
DEFAULT_OUT_ROOT = PATHS.samples_dir / "examples"
EVALUATION_INPUT_FILENAME = "evaluation_input.json"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--index", type=int, help="position in the sample file")
    which.add_argument("--example-id", help="exact example_id from the sample file")
    ap.add_argument("--grid-config", default=DEFAULT_GRID_CONFIG.name,
                    choices=[c.name for c in CANDIDATE_CONFIGS],
                    help="grid density candidate to render (see docs/grid_experiments.md)")
    ap.add_argument("--scope", choices=["segment", "episode"], default="segment",
                    help="segment ± pad (evaluator's view) or the whole episode")
    ap.add_argument("--instruction", default=DEFAULT_EVALUATION_INSTRUCTION,
                    help="override the placeholder evaluation instruction")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--print-json", action="store_true",
                    help="also dump the EvaluationInput JSON to stdout")
    args = ap.parse_args()

    examples = read_sample_jsonl(args.sample)
    if args.example_id is not None:
        matches = [e for e in examples if e.example_id == args.example_id]
        if not matches:
            raise SystemExit(f"no example {args.example_id!r} in {args.sample}")
        example = matches[0]
    else:
        example = examples[args.index]

    video_path = example.video.local_path()
    if not video_path.is_file():
        raise SystemExit(f"video not found at {video_path} — is the mount up?")

    grid_config = next(c for c in CANDIDATE_CONFIGS if c.name == args.grid_config)
    out_dir = args.out_root / example.example_id.replace("/", "__")
    eval_input, metrics = prepare_evaluation_input(
        example, video_path, out_dir,
        grid_config=grid_config, scope=args.scope, instruction=args.instruction,
    )
    write_evaluation_input(eval_input, out_dir / EVALUATION_INPUT_FILENAME)

    missing = missing_visual_files(eval_input, out_dir)
    if missing:
        raise SystemExit(f"referenced images missing after write: {missing}")

    seg = example.segment
    plan = eval_input.plan
    print(f"example      {example.example_id}")
    print(f"instruction  {example.instruction}")
    print(f"subtask      {seg.description}")
    print(f"segment      {seg.start_sec:.1f}-{seg.end_sec:.1f}s of episode "
          f"({seg.duration_sec:.1f}s; episode {example.video.episode_duration_sec:.1f}s)")
    print(f"scope/config {plan.scope} / {grid_config.name}")
    print(f"chunk window {plan.window_start_chunk_sec:.2f}-{plan.window_end_chunk_sec:.2f}s"
          f"  (segment at badge {plan.segment_badge_start_sec:.2f}"
          f"-{plan.segment_badge_end_sec:.2f}s)")
    print(f"images       {len(eval_input.visual_inputs)} grid(s), "
          f"{metrics.n_frames_extracted} frames, ~{metrics.est_visual_tokens} visual tokens")
    for v in eval_input.visual_inputs:
        print(f"             {v.path}  badges {v.badge_start_sec:.2f}-{v.badge_end_sec:.2f}s")
    print(f"eval instr   {eval_input.evaluation_instruction[:72]}...")
    print(f"wrote        {out_dir / EVALUATION_INPUT_FILENAME}")
    if args.print_json:
        print(eval_input.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
