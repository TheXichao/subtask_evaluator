"""The team's pre-review subtask format, as it exists on disk.

One row of ``meta/motion_desc_pipeline/open_vocab_subtasks.jsonl`` = one subtask
of one episode, produced by the team's subtasker (Qwen3.6-27B-sft over the 2 Hz
episode video) and rule-normalized. This file is written BEFORE human review;
the human-edited copy is ``open_vocab_subtasks_results_normalized_for_review.json``,
which this project deliberately does not read at this stage.

This module is the only place that knows the team's field names. Everything
downstream works with the canonical models in :mod:`subtask_checker.data.schema`.
"""

from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field

from subtask_checker import config


class SourceClock(BaseModel):
    """The redundant human-readable MM:SS rendering of the boundary times."""

    model_config = ConfigDict(extra="ignore")

    start: str
    end: str


class SourceSubtaskRow(BaseModel):
    """One JSONL row, with the team's own field names.

    Time bases (verified against the pipeline code and footage):
      * ``start_time`` / ``end_time`` — seconds relative to the EPISODE start,
        whole-second resolution (they round-trip through MM:SS strings).
      * ``source_from_timestamp`` / ``source_to_timestamp`` — where this episode's
        footage begins/ends inside the concatenated chunk mp4 ``source_chunk_video``,
        for THIS row's camera (``video_key``) only. LeRobot v3 splits chunk files by
        size, so these offsets are NOT valid for any other camera stream.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    dataset: str
    task: str
    goal: str
    episode_id: str
    episode_index: int
    step_index: int
    subtask: str
    video_key: str
    start_time: float
    end_time: float
    duration: float | None = None
    timestamps: SourceClock | None = None
    source_chunk_video: str
    source_from_timestamp: float
    source_to_timestamp: float | None = None
    status: str = "ok"
    generator_model: str | None = Field(default=None, alias="model")


def load_source_rows(jsonl_path: str | Path) -> list[SourceSubtaskRow]:
    """Parse one task's pre-review JSONL. Raises on malformed rows: the files are
    machine-written and uniform, so a parse failure means our schema assumption
    broke and should be surfaced, not skipped."""
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(SourceSubtaskRow.model_validate_json(line))
            except Exception as e:
                raise ValueError(f"{jsonl_path}:{line_no}: {e}") from e
    return rows


def iter_task_jsonls(data_root: Path | None = None) -> Iterator[tuple[str, Path]]:
    """Yield ``(task_name, jsonl_path)`` for every real task dataset that has
    pre-review subtasker output, in sorted task order."""
    root = data_root if data_root is not None else config.PATHS.data_root
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if task_dir.name.startswith(config.EXCLUDED_TASK_PREFIXES):
            continue
        jsonl = task_dir / config.SUBTASKS_JSONL_RELPATH
        if jsonl.is_file():
            yield task_dir.name, jsonl
