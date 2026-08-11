import shutil
import subprocess

import pytest

from subtask_checker.data.prepare import to_task_example
from subtask_checker.video.frames import compute_window
from subtask_checker.video.gridset import (
    CANDIDATE_CONFIGS,
    GridConfig,
    build_grid_set,
    estimate_visual_tokens,
    plan_grid_set,
)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def example(row):
    # conftest row: segment 0-10 s, episode spans 217.27-256.60 s in the chunk
    return to_task_example(row, "f", 0)


@pytest.fixture
def config():
    return GridConfig(name="t_5x5", rows=5, columns=5, sample_fps=2.0, tile_width=256)


def test_config_derived_quantities(config):
    assert config.frames_per_grid == 25
    assert config.sample_interval_sec == pytest.approx(0.5)
    assert config.chunk_duration_sec == pytest.approx(12.5)


def test_candidate_configs_have_unique_names():
    names = [c.name for c in CANDIDATE_CONFIGS]
    assert len(names) == len(set(names))


def test_compute_window_full_episode(example):
    start, end = compute_window(example, full_episode=True)
    assert start == pytest.approx(example.video.episode_start_in_chunk_sec)
    assert end == pytest.approx(example.video.episode_end_in_chunk_sec)


def test_compute_window_full_episode_requires_end(example):
    open_ended = example.model_copy(update={
        "video": example.video.model_copy(update={"episode_end_in_chunk_sec": None})
    })
    with pytest.raises(ValueError):
        compute_window(open_ended, full_episode=True)


def test_plan_uniform_sampling_and_chunking(example, config):
    plan = plan_grid_set(example, config, scope="segment")
    # window: segment 0-10 s clamped at episode start, +3 s trailing pad -> 13 s
    assert plan.window_end_chunk_sec - plan.window_start_chunk_sec == pytest.approx(13.0)
    # 13 s at 2 fps inclusive -> 27 frames -> 25 + 2
    assert plan.n_frames_planned == 27
    assert [len(g.sample_times_chunk_sec) for g in plan.grids] == [25, 2]
    assert [g.grid_index for g in plan.grids] == [0, 1]


def test_plan_badges_share_one_window_clock(example, config):
    plan = plan_grid_set(example, config, scope="segment")
    all_badges = [t for g in plan.grids for t in g.badge_times_sec]
    assert all_badges == [pytest.approx(k * 0.5) for k in range(len(all_badges))]
    # badge times continue across the grid boundary, no per-grid reset
    assert plan.grids[1].badge_times_sec[0] == pytest.approx(12.5)
    for g in plan.grids:
        for chunk_t, badge_t in zip(g.sample_times_chunk_sec, g.badge_times_sec):
            assert chunk_t - plan.window_start_chunk_sec == pytest.approx(badge_t, abs=1e-3)


def test_plan_segment_badges(example, config):
    plan = plan_grid_set(example, config, scope="segment")
    assert plan.segment_badge_start_sec == pytest.approx(0.0)  # clamped at episode start
    assert plan.segment_badge_end_sec == pytest.approx(10.0)


def test_plan_episode_scope_covers_episode(example, config):
    plan = plan_grid_set(example, config, scope="episode")
    assert plan.window_start_chunk_sec == pytest.approx(217.267, abs=1e-3)
    assert plan.window_end_chunk_sec == pytest.approx(256.6, abs=1e-3)
    # segment badges become episode-relative times
    assert plan.segment_badge_start_sec == pytest.approx(0.0)
    assert plan.segment_badge_end_sec == pytest.approx(10.0)
    # 39.33 s at 2 fps -> 79 frames -> 25+25+25+4
    assert [len(g.sample_times_chunk_sec) for g in plan.grids] == [25, 25, 25, 4]


def test_plan_rejects_exploding_frame_count(example):
    dense = GridConfig(name="t_dense", rows=10, columns=10, sample_fps=30.0, tile_width=64)
    with pytest.raises(ValueError, match="frames"):
        plan_grid_set(example, dense, scope="episode")


def test_estimate_visual_tokens():
    # 1280x960 -> ceil(1280/28)=46, ceil(960/28)=35
    assert estimate_visual_tokens(1280, 960) == 46 * 35
    assert estimate_visual_tokens(28, 28) == 1
    assert estimate_visual_tokens(29, 28) == 2


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "test.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=duration=20:size=640x480:rate=10",
         "-y", str(path)],
        check=True,
    )
    return path


@pytest.fixture
def synthetic_example(example):
    """Example remapped onto the 20 s synthetic clip: episode 0-20 s, segment 5-10 s."""
    return example.model_copy(update={
        "segment": example.segment.model_copy(update={"start_sec": 5.0, "end_sec": 10.0}),
        "video": example.video.model_copy(update={
            "episode_start_in_chunk_sec": 0.0,
            "episode_end_in_chunk_sec": 20.0,
        }),
    })


@needs_ffmpeg
def test_build_grid_set_end_to_end(synthetic_video, synthetic_example):
    config = GridConfig(name="t_2x3", rows=2, columns=3, sample_fps=1.0, tile_width=160)
    plan = plan_grid_set(synthetic_example, config, scope="segment")
    # window 2-13 s -> 12 frames at 1 fps -> two full 6-frame grids
    built, metrics = build_grid_set(synthetic_video, synthetic_example, plan)
    assert [g.grid_index for g in built] == [0, 1]
    assert all(g.jpeg.startswith(b"\xff\xd8") for g in built)
    assert metrics.n_frames_planned == 12
    assert metrics.n_frames_extracted == 12
    assert metrics.n_grids == 2
    assert (metrics.tile_width, metrics.tile_height) == (160, 120)
    assert (metrics.full_grid_width, metrics.full_grid_height) == (480, 240)
    assert (metrics.source_width, metrics.source_height) == (640, 480)
    assert metrics.source_fps == pytest.approx(10.0)
    assert metrics.downscale_vs_source == pytest.approx(0.25)
    assert metrics.episode_duration_sec == pytest.approx(20.0)
    assert metrics.est_visual_tokens == 2 * estimate_visual_tokens(480, 240)


@needs_ffmpeg
def test_build_grid_set_nothing_extractable(synthetic_video, example):
    # the real example's chunk times are far past the 20 s synthetic clip
    config = GridConfig(name="t_2x2", rows=2, columns=2, sample_fps=1.0, tile_width=160)
    plan = plan_grid_set(example, config, scope="segment")
    with pytest.raises(ValueError, match="no frames"):
        build_grid_set(synthetic_video, example, plan)
