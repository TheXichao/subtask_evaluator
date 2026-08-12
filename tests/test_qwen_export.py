import json
import math
import shutil
import subprocess

import pytest
from pydantic import ValidationError

from subtask_checker.data.prepare import to_task_example
from subtask_checker.eval_input import build_evaluation_input
from subtask_checker.qwen_export import (
    IMAGE_TAG,
    PLACEHOLDER_TARGET,
    QwenSample,
    QwenTarget,
    QwenTurn,
    build_human_turn,
    export_qwen_dataset,
    llama_factory_snippet,
    missing_media_files,
    read_annotations,
    registration_snippet,
    to_qwen_sample,
)
from subtask_checker.video.grid import GridMeta
from subtask_checker.video.gridset import BuiltGrid, GridConfig, plan_grid_set

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def example(row):
    return to_task_example(row, "f", 0)


@pytest.fixture
def config():
    return GridConfig(name="t_2x3", rows=2, columns=3, sample_fps=1.0, tile_width=160)


def _fake_built(grid_plan, config):
    n = len(grid_plan.badge_times_sec)
    rows = math.ceil(n / config.columns)
    tile_h = 120
    meta = GridMeta(
        columns=config.columns, rows=rows,
        tile_width=config.tile_width, tile_height=tile_h,
        width=config.columns * config.tile_width, height=rows * tile_h,
        n_tiles=n, badge_times_sec=grid_plan.badge_times_sec,
    )
    return BuiltGrid(grid_index=grid_plan.grid_index, jpeg=b"\xff\xd8fake", meta=meta)


@pytest.fixture
def eval_input(example, config):
    # window 0-13 s at 1 fps -> 14 frames -> three 2x3 grids
    plan = plan_grid_set(example, config, scope="segment")
    built = [_fake_built(g, config) for g in plan.grids]
    relpaths = [f"grids/{config.name}/grid_{g.grid_index:02d}.jpg" for g in plan.grids]
    return build_evaluation_input(example, plan, built, relpaths)


# --- human turn -------------------------------------------------------------


def test_human_turn_has_one_tag_per_grid_in_order(eval_input):
    turn = build_human_turn(eval_input)
    assert turn.count(IMAGE_TAG) == len(eval_input.visual_inputs) == 3
    assert turn.startswith(IMAGE_TAG)  # tags lead, in list order


def test_human_turn_carries_instruction_and_segment_facts(eval_input):
    turn = build_human_turn(eval_input)
    assert eval_input.evaluation_instruction in turn
    assert eval_input.example.instruction in turn
    assert eval_input.example.segment.description in turn
    # times on the badge clock, not episode or chunk time
    assert f"{eval_input.plan.segment_badge_start_sec:.1f}s" in turn
    assert f"{eval_input.plan.segment_badge_end_sec:.1f}s" in turn


# --- record assembly --------------------------------------------------------


def test_sample_rebases_paths_and_pairs_turns(eval_input):
    sample = to_qwen_sample(eval_input, PLACEHOLDER_TARGET, media_subdir="ex__001")
    assert sample.image == [
        "ex__001/grids/t_2x3/grid_00.jpg",
        "ex__001/grids/t_2x3/grid_01.jpg",
        "ex__001/grids/t_2x3/grid_02.jpg",
    ]
    assert [t.role for t in sample.conversations] == ["human", "gpt"]
    assert sample.conversations[1].value == PLACEHOLDER_TARGET.text


def test_annotation_dict_uses_official_keys(eval_input):
    d = to_qwen_sample(eval_input, PLACEHOLDER_TARGET).to_annotation()
    assert set(d) == {"image", "conversations"}
    assert set(d["conversations"][0]) == {"from", "value"}
    assert d["conversations"][0]["from"] == "human"
    assert d["conversations"][1]["from"] == "gpt"
    # round-trips through the wire form
    assert QwenSample.model_validate(json.loads(json.dumps(d))).to_annotation() == d


def test_tag_count_must_match_image_count():
    with pytest.raises(ValidationError, match="one tag per image"):
        QwenSample(
            image=["a.jpg", "b.jpg"],
            conversations=[
                QwenTurn(role="human", value=f"{IMAGE_TAG}\nonly one tag"),
                QwenTurn(role="gpt", value="x"),
            ],
        )


def test_tag_in_assistant_turn_rejected():
    with pytest.raises(ValidationError, match="assistant"):
        QwenSample(
            image=["a.jpg"],
            conversations=[
                QwenTurn(role="human", value=f"{IMAGE_TAG} q"),
                QwenTurn(role="gpt", value=f"answer {IMAGE_TAG}"),
            ],
        )


def test_video_tag_rejected():
    with pytest.raises(ValidationError, match="video"):
        QwenSample(
            image=["a.jpg"],
            conversations=[
                QwenTurn(role="human", value=f"{IMAGE_TAG} <video> q"),
                QwenTurn(role="gpt", value="x"),
            ],
        )


def test_absolute_image_path_rejected():
    with pytest.raises(ValidationError, match="relative"):
        QwenSample(
            image=["/abs/a.jpg"],
            conversations=[
                QwenTurn(role="human", value=f"{IMAGE_TAG} q"),
                QwenTurn(role="gpt", value="x"),
            ],
        )


def test_roles_must_alternate_starting_human():
    with pytest.raises(ValidationError, match="alternate"):
        QwenSample(
            image=["a.jpg"],
            conversations=[
                QwenTurn(role="gpt", value="x"),
                QwenTurn(role="human", value=f"{IMAGE_TAG} q"),
            ],
        )


def test_target_requires_source_and_bans_ground_truth():
    with pytest.raises(ValidationError):
        QwenTarget(text="x", source="")
    with pytest.raises(ValidationError, match="ground_truth"):
        QwenTarget(text="x", source="ground_truth")


# --- export end to end (synthetic clip, no mount) ---------------------------


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
def synthetic_examples(example, synthetic_video, monkeypatch):
    """Two examples remapped onto the 20 s synthetic clip."""
    from subtask_checker.data.schema import VideoRef

    monkeypatch.setattr(
        VideoRef, "local_path", lambda self: synthetic_video, raising=True
    )
    base = example.model_copy(update={
        "segment": example.segment.model_copy(update={"start_sec": 5.0, "end_sec": 10.0}),
        "video": example.video.model_copy(update={
            "episode_start_in_chunk_sec": 0.0,
            "episode_end_in_chunk_sec": 20.0,
        }),
    })
    second = base.model_copy(update={
        "example_id": base.example_id.replace("step_000", "step_001"),
        "step_index": 1,
        "segment": base.segment.model_copy(update={"start_sec": 10.0, "end_sec": 15.0}),
    })
    return [base, second]


@needs_ffmpeg
def test_export_end_to_end(synthetic_examples, config, tmp_path):
    out = tmp_path / "qwen" / "dev2"
    manifest = export_qwen_dataset(
        synthetic_examples, out, name="dev2", grid_config=config
    )

    assert len(manifest.records) == 2
    assert manifest.skipped_missing_video == []
    assert manifest.target_sources == ["placeholder_pending_labels"]
    assert manifest.records[0].annotation_source == "subtasker_pre_review"
    assert manifest.max_grid_pixels > 0

    samples = read_annotations(out)
    assert len(samples) == 2
    assert missing_media_files(out) == []
    for s in samples:
        assert s.conversations[1].value == PLACEHOLDER_TARGET.text
    # per-example evaluation_input.json is inspectable next to the grids
    assert (out / "media" / synthetic_examples[0].example_id.replace("/", "__")
            / "evaluation_input.json").is_file()
    # manifest round-trips
    from subtask_checker.qwen_export import QwenDatasetManifest
    assert QwenDatasetManifest.model_validate_json(
        (out / "manifest.json").read_text()
    ) == manifest


@needs_ffmpeg
def test_export_with_real_targets(synthetic_examples, config, tmp_path):
    targets = {
        ex.example_id: QwenTarget(text=f"verdict for {ex.step_index}", source="human_review_v0")
        for ex in synthetic_examples
    }
    manifest = export_qwen_dataset(
        synthetic_examples, tmp_path / "d", name="d", targets=targets, grid_config=config
    )
    assert manifest.target_sources == ["human_review_v0"]
    samples = read_annotations(tmp_path / "d")
    assert samples[0].conversations[1].value == "verdict for 0"
    assert samples[1].conversations[1].value == "verdict for 1"


def test_export_partial_targets_rejected(synthetic_examples, config, tmp_path):
    only_first = {
        synthetic_examples[0].example_id: QwenTarget(text="x", source="human_review_v0")
    }
    with pytest.raises(ValueError, match="targets missing"):
        export_qwen_dataset(
            synthetic_examples, tmp_path / "d", name="d",
            targets=only_first, grid_config=config,
        )


def test_export_duplicate_ids_rejected(synthetic_examples, config, tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        export_qwen_dataset(
            [synthetic_examples[0], synthetic_examples[0]],
            tmp_path / "d", name="d", grid_config=config,
        )


def test_export_bad_name_rejected(synthetic_examples, config, tmp_path):
    with pytest.raises(ValueError, match="name"):
        export_qwen_dataset(
            synthetic_examples, tmp_path / "d", name="bad name!", grid_config=config
        )


def test_export_all_videos_missing_fails_loudly(example, config, tmp_path):
    missing = example.model_copy(update={
        "video": example.video.model_copy(
            update={"chunk_video_server_path": "/nonexistent/nowhere.mp4"}
        )
    })
    with pytest.raises(ValueError, match="nothing to export"):
        export_qwen_dataset([missing], tmp_path / "d", name="d", grid_config=config)


def test_registration_snippet_points_at_export(tmp_path):
    snippet = registration_snippet("dev2", tmp_path / "dev2")
    assert '"annotation_path"' in snippet and "annotations.json" in snippet
    assert '"data_path"' in snippet and "media" in snippet
    assert 'data_dict["dev2"] = DEV2' in snippet


def test_llama_factory_snippet_registers_same_file(tmp_path):
    entry = json.loads(llama_factory_snippet("dev2", tmp_path / "dev2"))["dev2"]
    assert entry["file_name"].endswith("dev2/annotations.json")
    assert entry["formatting"] == "sharegpt"
    # LLaMA-Factory's ShareGPT tag defaults (from/value, human/gpt) match our
    # records, so only the image column needs remapping — no "tags" key.
    assert entry["columns"] == {"messages": "conversations", "images": "image"}
    assert "tags" not in entry


def test_records_satisfy_llama_factory_sharegpt_rules(eval_input):
    # LF silently DROPS samples with odd message counts or user turns at odd
    # indices; pin that our records can never trip that.
    sample = to_qwen_sample(eval_input, PLACEHOLDER_TARGET)
    assert len(sample.conversations) % 2 == 0
    for i, turn in enumerate(sample.conversations):
        assert turn.role == ("human" if i % 2 == 0 else "gpt")
    # images is always a list (never a bare string): uniform Arrow schema
    assert isinstance(sample.to_annotation()["image"], list)
