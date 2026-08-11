import io
import shutil
import subprocess

import pytest
from PIL import Image

from subtask_checker.data.prepare import to_task_example
from subtask_checker.video.frames import ExtractedFrame, extract_frame, extract_frames, plan_frames
from subtask_checker.video.grid import build_grid

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def example(row):
    return to_task_example(row, "f", 0)


def test_plan_covers_segment_with_pad(example):
    plan = plan_frames(example)
    seg_start, seg_end = example.segment_chunk_window()
    # start of episode: the leading pad must clamp to the episode boundary,
    # never leak into the previous episode in the chunk file
    assert plan.window_start_chunk_sec == pytest.approx(seg_start, abs=1e-3)
    assert plan.window_end_chunk_sec == pytest.approx(seg_end + 3.0, abs=1e-3)
    assert plan.segment_badge_start_sec == pytest.approx(0.0)
    assert plan.segment_badge_end_sec == pytest.approx(10.0)


def test_plan_clamps_to_episode_end(example):
    late = example.model_copy(update={
        "segment": example.segment.model_copy(update={"start_sec": 30.0, "end_sec": 39.0})
    })
    plan = plan_frames(late)
    assert plan.window_end_chunk_sec <= example.video.episode_end_in_chunk_sec + 1e-6


def test_plan_respects_frame_budget_and_min_interval(example):
    plan = plan_frames(example, max_frames=12, min_interval_sec=0.5)
    assert len(plan.sample_times_chunk_sec) <= 12
    assert plan.sample_interval_sec >= 0.5
    assert len(plan.badge_times_sec) == len(plan.sample_times_chunk_sec)
    assert plan.badge_times_sec[0] == pytest.approx(0.0)
    # badge times are the chunk times rebased to the window start
    for chunk_t, badge_t in zip(plan.sample_times_chunk_sec, plan.badge_times_sec):
        assert chunk_t - plan.window_start_chunk_sec == pytest.approx(badge_t, abs=1e-3)


def test_plan_short_segment_uses_min_interval(example):
    short = example.model_copy(update={
        "segment": example.segment.model_copy(update={"start_sec": 4.0, "end_sec": 5.0})
    })
    plan = plan_frames(short, context_pad_sec=1.0, min_interval_sec=0.5)
    assert plan.sample_interval_sec == pytest.approx(0.5)


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "test.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=10",
         "-y", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
def test_extract_frame_returns_jpeg(synthetic_video):
    data = extract_frame(synthetic_video, 1.0)
    assert data is not None and data.startswith(b"\xff\xd8")
    assert Image.open(io.BytesIO(data)).size == (320, 240)


@needs_ffmpeg
def test_extract_frame_past_end_is_none(synthetic_video):
    assert extract_frame(synthetic_video, 99.0) is None


@needs_ffmpeg
def test_extract_frames_drops_unreadable(synthetic_video, example):
    plan = plan_frames(example)
    # chunk times of the real example are far past this 4 s synthetic clip
    assert extract_frames(synthetic_video, plan) == []


def _fake_frame(badge_sec, color):
    img = Image.new("RGB", (640, 480), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return ExtractedFrame(badge_time_sec=badge_sec, chunk_time_sec=badge_sec, jpeg=buf.getvalue())


def test_build_grid_geometry():
    frames = [_fake_frame(i * 0.5, ("red", "green", "blue")[i % 3]) for i in range(10)]
    jpeg, meta = build_grid(frames, columns=4, tile_width=336)
    assert (meta.rows, meta.columns, meta.n_tiles) == (3, 4, 10)
    assert meta.tile_height == 252  # 480 * 336/640
    img = Image.open(io.BytesIO(jpeg))
    assert img.size == (4 * 336, 3 * 252) == (meta.width, meta.height)
    assert meta.badge_times_sec == [i * 0.5 for i in range(10)]


def test_build_grid_rejects_empty():
    with pytest.raises(ValueError):
        build_grid([])
