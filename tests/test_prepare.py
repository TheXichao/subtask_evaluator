import pytest

from subtask_checker.data.prepare import (
    read_sample_jsonl,
    select_stratified,
    to_task_example,
    write_sample_jsonl,
)
from subtask_checker.data.source import SourceSubtaskRow


def test_to_task_example_maps_fields(row):
    ex = to_task_example(row, source_file="/srv/open_vocab_subtasks.jsonl", row_index=0)
    assert ex.example_id == "410_g1_ir/clear_plates/episode_000017/step_000"
    assert ex.instruction.startswith("Place the plates")
    assert ex.segment.start_sec == 0.0 and ex.segment.end_sec == 10.0
    assert ex.segment.duration_sec == 10.0
    assert ex.video.episode_start_in_chunk_sec == pytest.approx(217.267, abs=1e-3)
    assert ex.video.episode_duration_sec == pytest.approx(39.33, abs=1e-2)
    assert ex.provenance.annotation_source == "subtasker_pre_review"
    assert ex.provenance.source_row_index == 0


def test_segment_chunk_window(row):
    ex = to_task_example(row, "f", 0)
    start, end = ex.segment_chunk_window()
    assert start == pytest.approx(217.267, abs=1e-3)
    assert end == pytest.approx(227.267, abs=1e-3)


def test_rejects_non_ok_status(row_dict):
    row = SourceSubtaskRow.model_validate({**row_dict, "status": "failed"})
    with pytest.raises(ValueError, match="status"):
        to_task_example(row, "f", 0)


def test_rejects_degenerate_segment(row_dict):
    row = SourceSubtaskRow.model_validate({**row_dict, "end_time": 0.0})
    with pytest.raises(ValueError, match="degenerate"):
        to_task_example(row, "f", 0)


def test_rejects_end_time_past_episode(row_dict):
    row = SourceSubtaskRow.model_validate({**row_dict, "end_time": 45.0})
    with pytest.raises(ValueError, match="exceeds episode span"):
        to_task_example(row, "f", 0)


def _examples(row, task, n):
    out = []
    for i in range(n):
        ex = to_task_example(row, "f", i)
        out.append(ex.model_copy(update={
            "task": task,
            "example_id": f"410_g1_ir/{task}/{ex.episode_id}/step_{i:03d}",
            "step_index": i,
        }))
    return out


def test_select_stratified_spreads_and_is_deterministic(row):
    pool = {t: _examples(row, t, 10) for t in ("task_a", "task_b", "task_c")}
    picked = select_stratified(pool, count=6, seed=42)
    assert len(picked) == 6
    assert {e.task for e in picked} == {"task_a", "task_b", "task_c"}
    again = select_stratified(pool, count=6, seed=42)
    assert [e.example_id for e in picked] == [e.example_id for e in again]
    other_seed = select_stratified(pool, count=6, seed=7)
    assert [e.example_id for e in picked] != [e.example_id for e in other_seed]


def test_select_stratified_task_change_keeps_other_tasks_stable(row):
    pool = {t: _examples(row, t, 10) for t in ("task_a", "task_b")}
    with_c = dict(pool, task_c=_examples(row, "task_c", 10))
    base = [e.example_id for e in select_stratified(pool, 4, seed=1) if e.task == "task_a"]
    plus = [e.example_id for e in select_stratified(with_c, 6, seed=1) if e.task == "task_a"]
    assert plus[: len(base)] == base


def test_select_stratified_exhausts_gracefully(row):
    pool = {"task_a": _examples(row, "task_a", 2)}
    assert len(select_stratified(pool, count=10, seed=0)) == 2


def test_sample_jsonl_roundtrip(tmp_path, row):
    examples = _examples(row, "clear_plates", 3)
    p = tmp_path / "sample.jsonl"
    write_sample_jsonl(examples, p)
    back = read_sample_jsonl(p)
    assert back == examples
