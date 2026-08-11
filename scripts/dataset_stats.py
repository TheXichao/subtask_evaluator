#!/usr/bin/env python3
"""Sweep /home/xichao/remote/410_g1_ir and collect per-task dataset statistics.

Read-only over the sshfs mount. Reads only small metadata files (info.json,
downsample.json, memory/*.json, motion_desc_pipeline summaries) — never videos
or data parquets.
"""
import json
import os
import statistics
import sys

from subtask_checker.config import PATHS

ROOT = str(PATHS.data_root)
OUT = str(PATHS.project_root / "docs" / "dataset_stats.json")


def safe_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"__error__": str(e)}


def count_lines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def main():
    tasks = sorted(
        d for d in os.listdir(ROOT)
        if os.path.isdir(os.path.join(ROOT, d))
    )
    report = {}
    all_durations = []
    for t in tasks:
        tdir = os.path.join(ROOT, t)
        meta = os.path.join(tdir, "meta")
        entry = {"has_meta": os.path.isdir(meta)}
        if not entry["has_meta"]:
            entry["top_level"] = sorted(os.listdir(tdir))[:20]
            report[t] = entry
            continue

        info = safe_json(os.path.join(meta, "info.json"))
        entry["codebase_version"] = info.get("codebase_version")
        entry["total_episodes"] = info.get("total_episodes")
        entry["total_frames"] = info.get("total_frames")
        entry["fps"] = info.get("fps")
        entry["splits"] = info.get("splits")
        feats = info.get("features", {})
        entry["video_keys"] = [k for k, v in feats.items() if isinstance(v, dict) and v.get("dtype") == "video"]
        if entry.get("total_frames") and entry.get("fps"):
            entry["total_hours"] = round(entry["total_frames"] / entry["fps"] / 3600, 2)

        ds = os.path.join(meta, "downsample.json")
        if os.path.exists(ds):
            sel = safe_json(ds)
            entry["downsample_selected"] = len(sel) if isinstance(sel, list) else None

        mem = os.path.join(meta, "memory")
        if os.path.isdir(mem):
            files = sorted(f for f in os.listdir(mem) if f.endswith(".json"))
            entry["memory_episodes"] = len(files)
            n_steps, durs, goals = [], [], set()
            fps_vals = set()
            for f in files:
                d = safe_json(os.path.join(mem, f))
                if "__error__" in d:
                    continue
                goals.add(d.get("goal"))
                steps = d.get("memory", [])
                n_steps.append(len(steps))
                for s in steps:
                    st, en = s.get("start_time"), s.get("end_time")
                    if st is not None and en is not None:
                        durs.append(en - st)
                    fps_vals.add(s.get("video_fps"))
            entry["goals"] = sorted(g for g in goals if g)
            entry["memory_video_fps"] = sorted(v for v in fps_vals if v is not None)
            if n_steps:
                entry["steps_per_episode"] = {
                    "total": sum(n_steps),
                    "mean": round(statistics.mean(n_steps), 2),
                    "min": min(n_steps),
                    "max": max(n_steps),
                }
            if durs:
                entry["subtask_duration_sec"] = {
                    "mean": round(statistics.mean(durs), 2),
                    "median": round(statistics.median(durs), 2),
                    "min": round(min(durs), 2),
                    "max": round(max(durs), 2),
                }
                all_durations.extend(durs)

        mdp = os.path.join(meta, "motion_desc_pipeline")
        if os.path.isdir(mdp):
            entry["mdp_files"] = sorted(os.listdir(mdp))
            entry["open_vocab_subtask_lines"] = count_lines(os.path.join(mdp, "open_vocab_subtasks.jsonl"))
            entry["motion_description_lines"] = count_lines(os.path.join(mdp, "motion_descriptions.jsonl"))
            ann = os.path.join(mdp, "annotations.json")
            if os.path.exists(ann):
                a = safe_json(ann)
                entry["annotations_records"] = len(a) if isinstance(a, list) else None
            acc = safe_json(os.path.join(mdp, "subtask_accuracy_summary.json"))
            if "__error__" not in acc:
                entry["accuracy"] = {
                    k: acc.get(k)
                    for k in (
                        "episodes", "model_steps_total", "human_steps_total",
                        "mean_time_segmentation_accuracy_iou",
                        "mean_subtask_content_accuracy_similarity",
                        "step_count_match_rate",
                    )
                }
        report[t] = entry
        print(f"done {t}", file=sys.stderr)

    if all_durations:
        report["__global__"] = {
            "all_subtask_count": len(all_durations),
            "duration_mean": round(statistics.mean(all_durations), 2),
            "duration_median": round(statistics.median(all_durations), 2),
            "duration_min": round(min(all_durations), 2),
            "duration_max": round(max(all_durations), 2),
        }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps(report.get("__global__", {}), indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
