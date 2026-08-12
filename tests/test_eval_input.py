import math
import shutil
import subprocess

import pytest
from pydantic import ValidationError

from subtask_checker.data.prepare import to_task_example
from subtask_checker.data.schema import Segment
from subtask_checker.eval_input import (
    DEFAULT_EVALUATION_INSTRUCTION,
    EvaluationInput,
    VisualInput,
    build_evaluation_input,
    missing_visual_files,
    prepare_evaluation_input,
    read_evaluation_input,
    write_evaluation_input,
)
from subtask_checker.video.grid import GridMeta
from subtask_checker.video.gridset import BuiltGrid, GridConfig, plan_grid_set

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def example(row):
    # conftest row: segment 0-10 s, episode spans 217.27-256.60 s in the chunk
    return to_task_example(row, "f", 0)


@pytest.fixture
def config():
    return GridConfig(name="t_2x3", rows=2, columns=3, sample_fps=1.0, tile_width=160)


@pytest.fixture
def plan(example, config):
    # window 0-13 s at 1 fps -> 14 frames -> grids of 6 + 6 + 2
    return plan_grid_set(example, config, scope="segment")


def _fake_built(grid_plan, config):
    """A BuiltGrid as build_grid_set would produce, without running ffmpeg."""
    n = len(grid_plan.badge_times_sec)
    rows = math.ceil(n / config.columns)
    tile_h = 120  # 480 * 160/640
    meta = GridMeta(
        columns=config.columns, rows=rows,
        tile_width=config.tile_width, tile_height=tile_h,
        width=config.columns * config.tile_width, height=rows * tile_h,
        n_tiles=n, badge_times_sec=grid_plan.badge_times_sec,
    )
    return BuiltGrid(grid_index=grid_plan.grid_index, jpeg=b"\xff\xd8fake", meta=meta)


@pytest.fixture
def built(plan, config):
    return [_fake_built(g, config) for g in plan.grids]


@pytest.fixture
def relpaths(plan, config):
    return [f"grids/{config.name}/grid_{g.grid_index:02d}.jpg" for g in plan.grids]


@pytest.fixture
def eval_input(example, plan, built, relpaths):
    return build_evaluation_input(example, plan, built, relpaths)


def test_valid_input_assembles(eval_input, example, config):
    assert eval_input.example == example
    assert eval_input.grid_config == config
    assert len(eval_input.visual_inputs) == 3
    assert eval_input.evaluation_instruction == DEFAULT_EVALUATION_INSTRUCTION
    assert [v.grid_index for v in eval_input.visual_inputs] == [0, 1, 2]


def test_segment_timestamps_propagate(eval_input):
    # segment 0-10 s of the episode, window clamped at the episode start
    assert eval_input.example.segment.start_sec == pytest.approx(0.0)
    assert eval_input.example.segment.end_sec == pytest.approx(10.0)
    assert eval_input.plan.segment_badge_start_sec == pytest.approx(0.0)
    assert eval_input.plan.segment_badge_end_sec == pytest.approx(10.0)


def test_images_share_one_badge_clock(eval_input):
    v = eval_input.visual_inputs
    assert v[0].badge_start_sec == pytest.approx(0.0)
    # each grid picks up exactly where the previous one left off (1 fps)
    assert v[1].badge_start_sec == pytest.approx(v[0].badge_end_sec + 1.0)
    assert v[2].badge_start_sec == pytest.approx(v[1].badge_end_sec + 1.0)
    assert v[-1].badge_end_sec == pytest.approx(13.0)


def test_single_image_fits_same_schema(example):
    # a grid big enough for the whole 14-frame window -> exactly one image
    big = GridConfig(name="t_4x4", rows=4, columns=4, sample_fps=1.0, tile_width=160)
    plan = plan_grid_set(example, big, scope="segment")
    built = [_fake_built(g, big) for g in plan.grids]
    ev = build_evaluation_input(example, plan, built, ["grids/t_4x4/grid_00.jpg"])
    assert len(ev.visual_inputs) == 1


def test_no_visual_inputs_rejected(example, plan):
    with pytest.raises(ValidationError):
        EvaluationInput(
            example=example, plan=plan, visual_inputs=[],
            evaluation_instruction=DEFAULT_EVALUATION_INSTRUCTION,
        )


def test_absolute_visual_path_rejected(built):
    with pytest.raises(ValidationError, match="relative"):
        VisualInput(path="/abs/grid_00.jpg", grid_index=0, meta=built[0].meta)


def test_empty_instruction_rejected(example, plan, built, relpaths):
    with pytest.raises(ValidationError):
        build_evaluation_input(example, plan, built, relpaths, instruction="")


def test_unknown_grid_index_rejected(example, plan, built):
    with pytest.raises(ValidationError, match="not in the plan"):
        build_evaluation_input(
            example, plan, [built[0].model_copy(update={"grid_index": 99})], ["g.jpg"]
        )


def test_unordered_grid_indices_rejected(example, plan, built, relpaths):
    with pytest.raises(ValidationError, match="ascending"):
        build_evaluation_input(example, plan, list(reversed(built)), relpaths)


def test_mismatched_path_count_raises(example, plan, built):
    with pytest.raises(ValueError, match="paths"):
        build_evaluation_input(example, plan, built, ["only_one.jpg"])


def test_degenerate_window_rejected(example, plan, built, relpaths):
    bad = plan.model_copy(update={"window_end_chunk_sec": plan.window_start_chunk_sec})
    with pytest.raises(ValidationError, match="window"):
        build_evaluation_input(example, bad, built, relpaths)


def test_invalid_segment_data_rejected():
    with pytest.raises(ValidationError, match="degenerate"):
        Segment(start_sec=5.0, end_sec=5.0, description="x")


def test_serialisation_round_trip(eval_input, tmp_path):
    assert EvaluationInput.model_validate_json(eval_input.model_dump_json()) == eval_input
    p = tmp_path / "evaluation_input.json"
    write_evaluation_input(eval_input, p)
    assert read_evaluation_input(p) == eval_input


def test_missing_visual_files_reported(eval_input, tmp_path):
    assert len(missing_visual_files(eval_input, tmp_path)) == 3
    for v in eval_input.visual_inputs:
        (tmp_path / v.path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / v.path).write_bytes(b"\xff\xd8fake")
    assert missing_visual_files(eval_input, tmp_path) == []


# --- end to end from one sample row, on a synthetic clip (no mount needed) ---

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
def test_prepare_from_sample_end_to_end(synthetic_video, synthetic_example, config, tmp_path):
    out_dir = tmp_path / "example"
    eval_input, metrics = prepare_evaluation_input(
        synthetic_example, synthetic_video, out_dir, grid_config=config
    )
    # window 2-13 s -> 12 frames at 1 fps -> two full 2x3 grids
    assert len(eval_input.visual_inputs) == 2
    assert metrics.n_frames_extracted == 12
    assert missing_visual_files(eval_input, out_dir) == []
    for v in eval_input.visual_inputs:
        assert (out_dir / v.path).read_bytes().startswith(b"\xff\xd8")
    # the claimed segment sits inside the rendered window on the badge clock
    assert eval_input.plan.segment_badge_start_sec == pytest.approx(3.0)
    assert eval_input.plan.segment_badge_end_sec == pytest.approx(8.0)

    json_path = out_dir / "evaluation_input.json"
    write_evaluation_input(eval_input, json_path)
    assert read_evaluation_input(json_path) == eval_input
