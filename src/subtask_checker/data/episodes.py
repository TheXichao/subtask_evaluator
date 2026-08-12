"""Episode-level view of the data: task JSONL → episodes → ordered subtasks.

The source JSONL is flat (one row per subtask). This module supplies the
hierarchy the data is actually about — a TASK contains EPISODES; an episode
has ONE goal instruction, ONE video window in the chunk file, and an ordered
list of subtask annotations. Those invariants were verified against the source
data (per-episode ``goal`` and chunk-video reference are constant; step indices
are unique and ascending), but :class:`EpisodeSubtasks` validates them rather
than assuming them, so source drift fails loudly here instead of producing
subtly mixed episodes downstream.

Nothing new is invented: an episode is composed of the same canonical
:class:`TaskExample` objects the rest of the pipeline uses, so per-subtask
processing (grids, evaluation inputs, export) applies unchanged to an
episode's members.
"""

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from pydantic import BaseModel, ConfigDict, Field, model_validator

from subtask_checker.data import source as source_mod
from subtask_checker.data.prepare import source_file_server_path, to_task_example
from subtask_checker.data.schema import TaskExample, VideoRef


class EpisodeSubtasks(BaseModel):
    """All of one episode's machine-generated subtasks, in step order.

    The identity fields are hoisted from the members and validated to agree —
    an "episode" whose rows disagree about the goal or the video window is a
    data error, never something to paper over.
    """

    model_config = ConfigDict(extra="forbid")

    dataset: str
    task: str
    episode_id: str
    episode_index: int
    instruction: str
    video: VideoRef
    subtasks: list[TaskExample] = Field(min_length=1)

    @model_validator(mode="after")
    def _members_agree(self) -> "EpisodeSubtasks":
        for ex in self.subtasks:
            claimed = (ex.dataset, ex.task, ex.episode_id, ex.episode_index)
            ours = (self.dataset, self.task, self.episode_id, self.episode_index)
            if claimed != ours:
                raise ValueError(f"subtask {ex.example_id} belongs to {claimed}, not {ours}")
            if ex.instruction != self.instruction:
                raise ValueError(
                    f"subtask {ex.example_id} carries a different goal instruction: "
                    f"{ex.instruction!r} != {self.instruction!r}"
                )
            if ex.video != self.video:
                raise ValueError(
                    f"subtask {ex.example_id} references a different video window "
                    f"than its episode"
                )
        steps = [ex.step_index for ex in self.subtasks]
        if steps != sorted(set(steps)):
            raise ValueError(f"step indices must be unique and ascending, got {steps}")
        return self

    @property
    def n_subtasks(self) -> int:
        return len(self.subtasks)


def _bucket(examples: Iterable[TaskExample]) -> dict[tuple[str, str, str], list[TaskExample]]:
    buckets: dict[tuple[str, str, str], list[TaskExample]] = defaultdict(list)
    for ex in examples:
        buckets[(ex.dataset, ex.task, ex.episode_id)].append(ex)
    return buckets


def _assemble(members: list[TaskExample]) -> EpisodeSubtasks:
    members = sorted(members, key=lambda ex: ex.step_index)
    first = members[0]
    return EpisodeSubtasks(
        dataset=first.dataset,
        task=first.task,
        episode_id=first.episode_id,
        episode_index=first.episode_index,
        instruction=first.instruction,
        video=first.video,
        subtasks=members,
    )


def group_by_episode(examples: Iterable[TaskExample]) -> list[EpisodeSubtasks]:
    """Group per-subtask examples into episodes, ordered by (task, episode_index).

    Strict: raises ValueError if any episode's members disagree about identity,
    goal, or video window. Use :func:`load_task_episodes` for the tolerant
    counting behaviour appropriate when scanning source files.
    """
    episodes = [_assemble(members) for members in _bucket(examples).values()]
    return sorted(episodes, key=lambda e: (e.dataset, e.task, e.episode_index))


def load_task_episodes(
    jsonl_path: str | Path, data_root: Path | None = None
) -> tuple[list[EpisodeSubtasks], dict]:
    """Load one task's pre-review JSONL as episodes.

    Same skip policy as ``build_sample``: rows that cannot become a usable
    example are counted, not fatal — and additionally an episode whose members
    violate the episode invariants is dropped and reported, so a scan stays
    honest about what it excluded. Returns ``(episodes, report)``.
    """
    jsonl = Path(jsonl_path)
    source_file = source_file_server_path(jsonl, data_root)
    report: dict = {"rows": 0, "invalid_rows": 0, "episodes": 0, "invalid_episodes": []}

    examples: list[TaskExample] = []
    for i, row in enumerate(source_mod.load_source_rows(jsonl)):
        report["rows"] += 1
        try:
            examples.append(to_task_example(row, source_file=source_file, row_index=i))
        except ValueError:
            report["invalid_rows"] += 1

    episodes = []
    for key, members in sorted(_bucket(examples).items()):
        try:
            episodes.append(_assemble(members))
        except ValueError:
            report["invalid_episodes"].append(key[2])
    episodes.sort(key=lambda e: (e.dataset, e.task, e.episode_index))
    report["episodes"] = len(episodes)
    return episodes, report


def iter_task_episodes(
    data_root: Path | None = None, tasks: list[str] | None = None
) -> Iterator[tuple[str, list[EpisodeSubtasks], dict]]:
    """Yield ``(task_name, episodes, report)`` for every task with subtasker
    output, in sorted task order — the full task → episode → subtask traversal."""
    for task, jsonl in source_mod.iter_task_jsonls(data_root):
        if tasks is not None and task not in tasks:
            continue
        episodes, report = load_task_episodes(jsonl, data_root)
        yield task, episodes, report
