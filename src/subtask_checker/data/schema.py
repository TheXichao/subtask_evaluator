"""Canonical data model: what one experimental example IS, independent of how the
team happens to store it.

A :class:`TaskExample` is one subtask of one episode, with everything needed to
(a) locate the exact video segment deterministically and (b) trace the example
back to its source row. Video paths are stored in the server's own absolute form
(the stable identifier that appears in the team data); they are translated to the
local mount only at the point of use via :func:`subtask_checker.config.resolve_server_path`.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from subtask_checker import config


class Segment(BaseModel):
    """The annotated span, in seconds relative to the episode start."""

    start_sec: float
    end_sec: float
    description: str

    @model_validator(mode="after")
    def _positive_span(self) -> "Segment":
        if self.end_sec <= self.start_sec:
            raise ValueError(f"degenerate segment: start={self.start_sec} end={self.end_sec}")
        return self

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


class VideoRef(BaseModel):
    """Where this episode's footage lives, for one camera stream.

    ``episode_start_in_chunk_sec``/``episode_end_in_chunk_sec`` bound the episode
    inside the concatenated chunk mp4. They are valid ONLY for ``video_key``:
    LeRobot v3 splits chunk files by size, so other cameras reach the same episode
    at a different file and offset (resolvable via ``meta/episodes/*.parquet``,
    not needed while we use the single camera the subtasker used).
    """

    video_key: str
    chunk_video_server_path: str
    episode_start_in_chunk_sec: float
    episode_end_in_chunk_sec: float | None = None

    def local_path(self) -> Path:
        return config.resolve_server_path(self.chunk_video_server_path)

    def to_chunk_time(self, episode_time_sec: float) -> float:
        """Map an episode-relative time to a timestamp in the chunk video."""
        return self.episode_start_in_chunk_sec + episode_time_sec

    @property
    def episode_duration_sec(self) -> float | None:
        if self.episode_end_in_chunk_sec is None:
            return None
        return self.episode_end_in_chunk_sec - self.episode_start_in_chunk_sec


class Provenance(BaseModel):
    """Where the example came from, so no example is ever untraceable and label
    sources can never get mixed up later (pre-review model output today; other
    sources, e.g. human-reviewed or synthetic, would get distinct values here)."""

    annotation_source: str = "subtasker_pre_review"
    source_file_server_path: str
    source_row_index: int
    generator_model: str | None = None


class TaskExample(BaseModel):
    """One subtask resolved to its episode, instruction, and video segment."""

    model_config = ConfigDict(extra="forbid")

    example_id: str  # "<dataset>/<task>/<episode_id>/step_<NNN>"
    dataset: str
    task: str
    episode_id: str
    episode_index: int
    step_index: int
    instruction: str
    segment: Segment
    video: VideoRef
    provenance: Provenance

    def segment_chunk_window(self) -> tuple[float, float]:
        """The segment's span in chunk-video seconds."""
        return (
            self.video.to_chunk_time(self.segment.start_sec),
            self.video.to_chunk_time(self.segment.end_sec),
        )


def make_example_id(dataset: str, task: str, episode_id: str, step_index: int) -> str:
    return f"{dataset}/{task}/{episode_id}/step_{step_index:03d}"
