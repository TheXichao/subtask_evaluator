#!/usr/bin/env python3
"""Select a small development sample from the team's pre-review subtask data.

Reads every task's open_vocab_subtasks.jsonl from the read-only mount, converts
rows to canonical TaskExamples, and writes a deterministic stratified sample.

    uv run python scripts/build_sample.py --count 20
"""

import argparse
from collections import Counter
from pathlib import Path

from subtask_checker.config import PATHS
from subtask_checker.data.prepare import build_sample, write_sample_jsonl

DEFAULT_OUT = PATHS.dev_sample_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=20, help="number of subtasks to select")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tasks", help="comma-separated task names (default: all tasks)")
    ap.add_argument("--data-root", type=Path, help="override the dataset mount")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    tasks = args.tasks.split(",") if args.tasks else None
    examples, report = build_sample(
        count=args.count, seed=args.seed, data_root=args.data_root, tasks=tasks
    )
    write_sample_jsonl(examples, args.out)

    print(f"scanned {report['tasks']} tasks, {report['rows']} rows "
          f"({report['invalid_rows']} invalid, {report['missing_video_rows']} missing video)")
    print(f"selected {report['selected']} of {report['candidates']} candidates "
          f"across {len(report['selected_tasks'])} tasks -> {args.out}")
    per_task = Counter(e.task for e in examples)
    for i, ex in enumerate(examples):
        print(f"  [{i:2d}] {ex.example_id}  {ex.segment.start_sec:.0f}-{ex.segment.end_sec:.0f}s"
              f"  {ex.segment.description[:60]}")
    print("per task:", dict(sorted(per_task.items())))


if __name__ == "__main__":
    main()
