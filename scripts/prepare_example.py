#!/usr/bin/env python3
"""Turn ONE sampled subtask into a self-contained, inspectable example directory.

    uv run python scripts/prepare_example.py --index 0
    uv run python scripts/prepare_example.py --example-id 410_g1_ir/clear_plates/episode_000017/step_000

Output under data/samples/examples/<example_id>/:
    example.json   canonical TaskExample + frame plan + grid metadata
    grid.jpg       contact sheet, time badges counting from the window start
    frames/        the individual sampled frames
"""

import argparse
import json
from pathlib import Path

from subtask_checker.config import PATHS, require_writable
from subtask_checker.data.prepare import read_sample_jsonl
from subtask_checker.video.frames import (
    CONTEXT_PAD_SEC,
    MAX_FRAMES,
    extract_frames,
    plan_frames,
)
from subtask_checker.video.grid import build_grid

DEFAULT_SAMPLE = PATHS.dev_sample_jsonl
DEFAULT_OUT_ROOT = PATHS.samples_dir / "examples"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--index", type=int, help="position in the sample file")
    which.add_argument("--example-id", help="exact example_id from the sample file")
    ap.add_argument("--pad", type=float, default=CONTEXT_PAD_SEC,
                    help="context seconds shown around the segment")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
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

    plan = plan_frames(example, context_pad_sec=args.pad, max_frames=args.max_frames)
    frames = extract_frames(video_path, plan)
    if not frames:
        raise SystemExit(f"no frames could be extracted from {video_path}")
    grid_jpeg, grid_meta = build_grid(frames)

    out_dir = require_writable(args.out_root / example.example_id.replace("/", "__"))
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("frame_*.jpg"):
        stale.unlink()
    (out_dir / "grid.jpg").write_bytes(grid_jpeg)
    for i, fr in enumerate(frames):
        (frames_dir / f"frame_{i:02d}_badge_{fr.badge_time_sec:06.2f}s.jpg").write_bytes(fr.jpeg)

    record = {
        "example": example.model_dump(),
        "resolved_video_path": str(video_path),
        "frame_plan": plan.model_dump(),
        "grid": grid_meta.model_dump(),
        "n_frames_extracted": len(frames),
    }
    (out_dir / "example.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))

    seg = example.segment
    print(f"example      {example.example_id}")
    print(f"instruction  {example.instruction}")
    print(f"subtask      {seg.description}")
    print(f"segment      {seg.start_sec:.1f}-{seg.end_sec:.1f}s of episode "
          f"({seg.duration_sec:.1f}s; episode {example.video.episode_duration_sec:.1f}s)")
    print(f"video        {video_path}")
    print(f"chunk window {plan.window_start_chunk_sec:.2f}-{plan.window_end_chunk_sec:.2f}s"
          f"  (segment at badge {plan.segment_badge_start_sec:.2f}-{plan.segment_badge_end_sec:.2f}s)")
    print(f"frames       {len(frames)} @ {plan.sample_interval_sec:.2f}s interval")
    print(f"grid         {grid_meta.width}x{grid_meta.height} "
          f"({grid_meta.rows}x{grid_meta.columns} tiles)")
    print(f"wrote        {out_dir}/")


if __name__ == "__main__":
    main()
