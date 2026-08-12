"""Source rows -> canonical examples, and small deterministic dev samples."""

import random
from pathlib import Path

from subtask_checker import config
from subtask_checker.data import source as source_mod
from subtask_checker.data.schema import (
    Provenance,
    Segment,
    TaskExample,
    VideoRef,
    make_example_id,
)
from subtask_checker.data.source import SourceSubtaskRow

# The MM:SS format quantizes boundaries to whole seconds, so the last subtask's
# end_time may legitimately overshoot the episode's real duration by up to a second.
END_TIME_TOLERANCE_SEC = 1.0


def source_file_server_path(jsonl_path: Path, data_root: Path | None = None) -> str:
    """Provenance form of a source JSONL path: the server's own absolute path —
    stable across machines and mount points, same convention as the chunk-video
    reference. Falls back to the literal path for files outside the data root
    (e.g. test fixtures)."""
    root = data_root if data_root is not None else config.PATHS.data_root
    try:
        return f"{config.SERVER_DATA_PREFIX}/{jsonl_path.relative_to(root)}"
    except ValueError:
        return str(jsonl_path)


def to_task_example(
    row: SourceSubtaskRow, source_file: str | Path, row_index: int
) -> TaskExample:
    """Adapt one team row to the canonical model.

    Raises ValueError when the row cannot be a usable example (failed generation,
    degenerate span, or timestamps inconsistent with the episode's footage window).
    """
    if row.status != "ok":
        raise ValueError(f"row status is {row.status!r}, not 'ok'")

    if row.source_to_timestamp is not None:
        episode_duration = row.source_to_timestamp - row.source_from_timestamp
        if row.end_time > episode_duration + END_TIME_TOLERANCE_SEC:
            raise ValueError(
                f"end_time {row.end_time}s exceeds episode span {episode_duration:.2f}s"
            )

    return TaskExample(
        example_id=make_example_id(row.dataset, row.task, row.episode_id, row.step_index),
        dataset=row.dataset,
        task=row.task,
        episode_id=row.episode_id,
        episode_index=row.episode_index,
        step_index=row.step_index,
        instruction=row.goal,
        segment=Segment(
            start_sec=row.start_time,
            end_sec=row.end_time,
            description=row.subtask,
        ),
        video=VideoRef(
            video_key=row.video_key,
            chunk_video_server_path=row.source_chunk_video,
            episode_start_in_chunk_sec=row.source_from_timestamp,
            episode_end_in_chunk_sec=row.source_to_timestamp,
        ),
        provenance=Provenance(
            source_file_server_path=str(source_file),
            source_row_index=row_index,
            generator_model=row.generator_model,
        ),
    )


def select_stratified(
    examples_by_task: dict[str, list[TaskExample]], count: int, seed: int
) -> list[TaskExample]:
    """Pick ``count`` examples spread evenly across tasks, deterministically.

    Round-robin over sorted task names, each task's rows shuffled with an rng
    seeded from (seed, task) — so adding or removing a task does not change which
    rows the remaining tasks contribute.
    """
    shuffled = {}
    for task in sorted(examples_by_task):
        rows = list(examples_by_task[task])
        random.Random(f"{seed}:{task}").shuffle(rows)
        shuffled[task] = rows

    picked: list[TaskExample] = []
    depth = 0
    while len(picked) < count:
        advanced = False
        for task in sorted(shuffled):
            rows = shuffled[task]
            if depth < len(rows):
                picked.append(rows[depth])
                advanced = True
                if len(picked) >= count:
                    break
        if not advanced:  # every task exhausted
            break
        depth += 1
    return picked


def build_sample(
    count: int,
    seed: int,
    data_root: Path | None = None,
    tasks: list[str] | None = None,
    require_video: bool = True,
) -> tuple[list[TaskExample], dict]:
    """Load every task's pre-review rows, convert, and select a small sample.

    Returns (examples, report). ``report`` counts what was skipped and why, so
    a quietly shrinking candidate pool is visible.
    """
    root = data_root if data_root is not None else config.PATHS.data_root
    by_task: dict[str, list[TaskExample]] = {}
    report = {"tasks": 0, "rows": 0, "invalid_rows": 0, "missing_video_rows": 0}
    checked_videos: dict[str, bool] = {}

    for task, jsonl in source_mod.iter_task_jsonls(root):
        if tasks is not None and task not in tasks:
            continue
        report["tasks"] += 1
        source_file = source_file_server_path(jsonl, root)
        for i, row in enumerate(source_mod.load_source_rows(jsonl)):
            report["rows"] += 1
            try:
                ex = to_task_example(row, source_file=source_file, row_index=i)
            except ValueError:
                report["invalid_rows"] += 1
                continue
            if require_video:
                key = ex.video.chunk_video_server_path
                if key not in checked_videos:
                    checked_videos[key] = ex.video.local_path().is_file()
                if not checked_videos[key]:
                    report["missing_video_rows"] += 1
                    continue
            by_task.setdefault(task, []).append(ex)

    examples = select_stratified(by_task, count, seed)
    report["candidates"] = sum(len(v) for v in by_task.values())
    report["selected"] = len(examples)
    report["selected_tasks"] = sorted({e.task for e in examples})
    return examples, report


def write_sample_jsonl(examples: list[TaskExample], out_path: str | Path) -> None:
    out = config.require_writable(Path(out_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")


def read_sample_jsonl(path: str | Path) -> list[TaskExample]:
    with open(path, encoding="utf-8") as f:
        return [TaskExample.model_validate_json(line) for line in f if line.strip()]
