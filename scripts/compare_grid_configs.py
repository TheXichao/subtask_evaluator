#!/usr/bin/env python3
"""Render the SAME example under several frame/grid densities, side by side.

    uv run python scripts/compare_grid_configs.py --index 0
    uv run python scripts/compare_grid_configs.py --index 0 --scope episode
    uv run python scripts/compare_grid_configs.py --example-id <id> --configs a_12f_tile336_1fps,d_64f_tile160_4fps
    uv run python scripts/compare_grid_configs.py --index 0 --experiment experiments/tile_floor.json

The point is the empirical question behind the token-efficiency constraint: how
many frames fit in one grid before the actions needed for evaluation stop being
recognisable? This script produces the evidence; it does not pick a winner.

Without --experiment it runs the built-in density candidates (optionally
filtered with --configs). --experiment loads a source-controlled override file
(see experiments/ and docs/configuration.md).

Output under data/samples/grid_experiments/<example_id>/<scope>/<experiment>/:
    <config_name>/grid_00.jpg ...   the rendered grids
    <config_name>/metrics.json      plan + GridSetMetrics for that config
    comparison.json                 experiment + all configs' metrics in one place
    index.html                      open locally to inspect configs side by side
"""

import argparse
import html
import json
from pathlib import Path

from subtask_checker.config import PATHS, require_writable
from subtask_checker.data.prepare import read_sample_jsonl
from subtask_checker.experiments import (
    ExperimentConfig,
    default_experiment,
    load_experiment,
)
from subtask_checker.video.gridset import build_grid_set, plan_grid_set

DEFAULT_SAMPLE = PATHS.dev_sample_jsonl
DEFAULT_OUT_ROOT = PATHS.samples_dir / "grid_experiments"


def pick_experiment(args) -> ExperimentConfig:
    if args.experiment:
        return load_experiment(args.experiment)
    experiment = default_experiment()
    if args.configs:
        names = [n.strip() for n in args.configs.split(",") if n.strip()]
        by_name = {c.name: c for c in experiment.grids}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise SystemExit(f"unknown config(s) {missing}; known: {sorted(by_name)}")
        experiment = experiment.model_copy(update={"grids": [by_name[n] for n in names]})
    return experiment


def write_index_html(
    out_dir: Path, example, scope: str, experiment: ExperimentConfig, results: list[dict]
) -> None:
    """One self-contained page, grids at natural pixel size so detail is judged
    honestly — browser zoom would hide exactly what we are trying to measure."""
    seg = example.segment
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(experiment.name)} — {html.escape(example.example_id)}</title>",
        "<style>body{font-family:sans-serif;margin:1rem;background:#222;color:#eee}"
        "section{margin-bottom:2rem}img{display:block;margin:4px 0;image-rendering:auto}"
        ".m{color:#9c9;font-size:0.9rem}</style>",
        f"<h1>{html.escape(example.example_id)} — scope: {scope} — "
        f"experiment: {html.escape(experiment.name)}</h1>",
        f"<p>instruction: {html.escape(example.instruction)}<br>"
        f"subtask: {html.escape(seg.description)}<br>"
        f"segment {seg.start_sec:.1f}–{seg.end_sec:.1f}s of episode<br>"
        f"{html.escape(experiment.description)}</p>",
    ]
    for r in results:
        m = r["metrics"]
        parts.append(
            f"<section><h2>{html.escape(m['config_name'])}</h2>"
            f"<p class='m'>{m['n_grids']} grid(s) · {m['n_frames_extracted']} frames · "
            f"{m['rows']}×{m['columns']} @ {m['sample_fps']:g} fps · "
            f"tile {m['tile_width']}×{m['tile_height']} px · "
            f"full grid {m['full_grid_width']}×{m['full_grid_height']} px · "
            f"~{m['est_visual_tokens']} visual tokens "
            f"(~{m['est_visual_tokens_per_window_sec']:g}/s) · "
            f"segment at badge {r['plan']['segment_badge_start_sec']:.2f}–"
            f"{r['plan']['segment_badge_end_sec']:.2f}s</p>"
        )
        for name in r["grid_files"]:
            parts.append(f"<img src='{html.escape(name)}' alt='{html.escape(name)}'>")
        parts.append("</section>")
    (out_dir / "index.html").write_text("\n".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--index", type=int, help="position in the sample file")
    which.add_argument("--example-id", help="exact example_id from the sample file")
    ap.add_argument("--scope", choices=["segment", "episode"], default="segment",
                    help="segment ± pad (evaluator's view) or the whole episode")
    ap.add_argument("--configs", help="comma-separated names from the built-in candidates")
    ap.add_argument("--experiment", type=Path,
                    help="experiment override file, e.g. experiments/tile_floor.json "
                         "(mutually exclusive with --configs)")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = ap.parse_args()
    if args.experiment and args.configs:
        ap.error("--experiment and --configs are mutually exclusive")

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

    experiment = pick_experiment(args)
    out_dir = require_writable(
        args.out_root / example.example_id.replace("/", "__") / args.scope / experiment.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for config in experiment.grids:
        plan = plan_grid_set(example, config, scope=args.scope)
        built, metrics = build_grid_set(video_path, example, plan)

        config_dir = out_dir / config.name
        config_dir.mkdir(exist_ok=True)
        for stale in config_dir.glob("grid_*.jpg"):
            stale.unlink()
        grid_files = []
        for g in built:
            name = f"grid_{g.grid_index:02d}.jpg"
            (config_dir / name).write_bytes(g.jpeg)
            grid_files.append(f"{config.name}/{name}")
        record = {
            "plan": plan.model_dump(),
            "metrics": metrics.model_dump(),
            "grids": [g.meta.model_dump() for g in built],
        }
        (config_dir / "metrics.json").write_text(json.dumps(record, indent=2))
        results.append({**record, "grid_files": grid_files})

        m = metrics
        print(f"{config.name:24s} {m.n_grids} grid(s)  {m.n_frames_extracted:3d} frames  "
              f"tile {m.tile_width}x{m.tile_height}  grid {m.full_grid_width}x{m.full_grid_height}  "
              f"~{m.est_visual_tokens} tok (~{m.est_visual_tokens_per_window_sec:g}/s)")

    comparison = {
        "experiment": experiment.model_dump(),
        "example_id": example.example_id,
        "scope": args.scope,
        "resolved_video_path": str(video_path),
        "example": example.model_dump(),
        "configs": [r["metrics"] for r in results],
    }
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
    write_index_html(out_dir, example, args.scope, experiment, results)
    print(f"\nwrote {out_dir}/  (open index.html to compare visually)")


if __name__ == "__main__":
    main()
