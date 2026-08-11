"""Frame sampling for one example's video segment.

Geometry adapted from the previous evaluation project's proven configuration
(``~/Dev/evaluation/code/refiner_frames.py``): a context pad of 3 s around the
annotated span so both boundaries are visible with their surroundings, at most 12
frames per example, never sampled finer than 0.5 s (the subtasker's own input was
2 Hz). One deliberate difference: the sampling window is clamped to the episode's
own span inside the chunk file — chunk mp4s concatenate many episodes back-to-back,
so an unclamped pad would silently show footage of a NEIGHBOURING episode.

Extraction uses ffmpeg with ``-ss`` before ``-i`` (accurate seek, and OpenCV often
fails on these AV1 streams). ffmpeg can exit 0 yet write an empty file when a seek
lands past the end of a stream, so outputs are checked for a JPEG magic number.
"""

import json
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel

from subtask_checker.data.schema import TaskExample

CONTEXT_PAD_SEC = 3.0
MAX_FRAMES = 12
MIN_INTERVAL_SEC = 0.5


class VideoProbe(BaseModel):
    """Actual properties of the source stream, so grid geometry decisions are
    grounded in the real resolution instead of assumptions."""

    width: int
    height: int
    fps: float
    codec: str


def probe_video(video_path: str | Path) -> VideoProbe | None:
    """ffprobe the first video stream; None if the file can't be probed."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate",
            "-of", "json", str(video_path),
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    try:
        stream = json.loads(proc.stdout)["streams"][0]
        num, _, den = stream["avg_frame_rate"].partition("/")
        return VideoProbe(
            width=stream["width"],
            height=stream["height"],
            fps=round(float(num) / float(den or 1), 3),
            codec=stream["codec_name"],
        )
    except (KeyError, IndexError, ValueError, ZeroDivisionError, json.JSONDecodeError):
        return None


def compute_window(
    example: TaskExample,
    context_pad_sec: float = CONTEXT_PAD_SEC,
    min_span_sec: float = MIN_INTERVAL_SEC,
    full_episode: bool = False,
) -> tuple[float, float]:
    """The chunk-video window to sample: segment ± pad, or the whole episode.

    Either way the window is clamped to the episode's own span in the chunk file —
    chunks concatenate episodes back-to-back, so anything wider would silently show
    footage of a NEIGHBOURING episode.
    """
    ep_start = example.video.episode_start_in_chunk_sec
    ep_end = example.video.episode_end_in_chunk_sec
    if full_episode:
        if ep_end is None:
            raise ValueError("episode end in chunk is unknown; cannot window the full episode")
        return ep_start, ep_end
    seg_start, seg_end = example.segment_chunk_window()
    window_start = max(ep_start, seg_start - context_pad_sec)
    window_end = seg_end + context_pad_sec
    if ep_end is not None:
        window_end = min(ep_end, window_end)
    window_end = max(window_end, window_start + min_span_sec)
    return window_start, window_end


class FramePlan(BaseModel):
    """Which timestamps to extract, in both time bases.

    ``badge`` times are seconds from the start of the sampling window — the numbers
    burned into the frames, chosen so the model never needs absolute-time arithmetic.
    """

    window_start_chunk_sec: float
    window_end_chunk_sec: float
    sample_interval_sec: float
    sample_times_chunk_sec: list[float]
    badge_times_sec: list[float]
    segment_badge_start_sec: float
    segment_badge_end_sec: float
    context_pad_sec: float


class ExtractedFrame(BaseModel):
    badge_time_sec: float
    chunk_time_sec: float
    jpeg: bytes


def plan_frames(
    example: TaskExample,
    context_pad_sec: float = CONTEXT_PAD_SEC,
    max_frames: int = MAX_FRAMES,
    min_interval_sec: float = MIN_INTERVAL_SEC,
) -> FramePlan:
    seg_start, seg_end = example.segment_chunk_window()
    window_start, window_end = compute_window(
        example, context_pad_sec=context_pad_sec, min_span_sec=min_interval_sec
    )

    span = window_end - window_start
    interval = max(min_interval_sec, span / max(1, max_frames - 1))

    times = []
    t = window_start
    while t <= window_end + 1e-6 and len(times) < max_frames:
        times.append(round(t, 3))
        t += interval

    def rebase(x: float) -> float:
        return round(x - window_start, 3) + 0.0  # + 0.0 folds -0.0 into 0.0

    return FramePlan(
        window_start_chunk_sec=round(window_start, 3),
        window_end_chunk_sec=round(window_end, 3),
        sample_interval_sec=round(interval, 3),
        sample_times_chunk_sec=times,
        badge_times_sec=[rebase(x) for x in times],
        segment_badge_start_sec=rebase(seg_start),
        segment_badge_end_sec=rebase(seg_end),
        context_pad_sec=context_pad_sec,
    )


def extract_frame(video_path: str | Path, chunk_time_sec: float) -> bytes | None:
    """One JPEG frame at the given chunk-video timestamp, or None if unreadable."""
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", f"{chunk_time_sec:.3f}",
                "-i", str(video_path),
                "-frames:v", "1", "-q:v", "2",
                "-y", tmp.name,
            ],
            capture_output=True,
        )
        if proc.returncode != 0:
            return None
        data = Path(tmp.name).read_bytes()
    if len(data) < 128 or not data.startswith(b"\xff\xd8"):
        return None
    return data


def extract_frames(video_path: str | Path, plan: FramePlan) -> list[ExtractedFrame]:
    """Extract the planned frames. Unreadable timestamps are dropped, not fatal —
    one truncated stream should cost a frame, not the example."""
    frames = []
    for chunk_t, badge_t in zip(plan.sample_times_chunk_sec, plan.badge_times_sec):
        jpeg = extract_frame(video_path, chunk_t)
        if jpeg is not None:
            frames.append(
                ExtractedFrame(badge_time_sec=badge_t, chunk_time_sec=chunk_t, jpeg=jpeg)
            )
    return frames
