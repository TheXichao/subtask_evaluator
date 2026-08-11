from pathlib import Path

import pytest

from subtask_checker import config
from subtask_checker.data.source import SourceSubtaskRow, load_source_rows


def test_parses_real_row_shape(row):
    assert row.episode_id == "episode_000017"
    assert row.start_time == 0.0 and row.end_time == 10.0
    assert row.video_key == "observation.images.head_stereo_left"
    assert row.timestamps.start == "00:00"
    # the team's "model" field maps onto our aliased name
    assert row.generator_model.endswith("Qwen3.6-27B-sft-merged")


def test_ignores_unmodelled_team_fields(row):
    assert not hasattr(row, "subtask_normalization_method")


def test_load_source_rows(jsonl_file):
    rows = load_source_rows(jsonl_file)
    assert [r.step_index for r in rows] == [0, 1]


def test_load_source_rows_surfaces_bad_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"dataset": "x"}\n')
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_source_rows(p)


def test_resolve_server_path(row):
    local = config.resolve_server_path(row.source_chunk_video, data_root=Path("/mnt/local"))
    assert str(local) == (
        "/mnt/local/clear_plates/videos/observation.images.head_stereo_left/chunk-000/file-001.mp4"
    )
    # non-server paths pass through untouched
    assert str(config.resolve_server_path("/elsewhere/a.mp4")) == "/elsewhere/a.mp4"
