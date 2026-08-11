import json

import pytest

from subtask_checker.data.source import SourceSubtaskRow

# A real (abridged) row from clear_plates open_vocab_subtasks.jsonl, including
# team fields we deliberately ignore, to pin the boundary behaviour.
ROW_DICT = {
    "dataset": "410_g1_ir",
    "task": "clear_plates",
    "goal": "Place the plates on the rack from right to left in the order of red, green, and blue.",
    "episode_id": "episode_000017",
    "episode_index": 17,
    "step_index": 0,
    "subtask": "left hand pick up plate rack from table",
    "next_step": "right hand place red plate on rack",
    "video_key": "observation.images.head_stereo_left",
    "start_time": 0.0,
    "end_time": 10.0,
    "timestamps": {"start": "00:00", "end": "00:10"},
    "duration": 10.0,
    "source_chunk_video": "/mnt/raid0/supeng_g1_data/410_g1_ir/clear_plates/videos/observation.images.head_stereo_left/chunk-000/file-001.mp4",
    "source_from_timestamp": 217.26666666666668,
    "source_to_timestamp": 256.6,
    "source_timestamps": {"start": "03:37.27", "end": "04:16.60"},
    "model": "/mnt/data/users/wangshuqi/models/Qwen3.6-27B-sft-merged",
    "status": "ok",
    "subtask_normalization_method": "rule_canonical_v2",
}


@pytest.fixture
def row_dict():
    return dict(ROW_DICT)


@pytest.fixture
def row(row_dict):
    return SourceSubtaskRow.model_validate(row_dict)


@pytest.fixture
def jsonl_file(tmp_path, row_dict):
    p = tmp_path / "open_vocab_subtasks.jsonl"
    p.write_text(json.dumps(row_dict) + "\n" + json.dumps({**row_dict, "step_index": 1}) + "\n")
    return p
