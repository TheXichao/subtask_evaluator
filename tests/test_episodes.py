import json

import pytest

from subtask_checker import config
from subtask_checker.data.episodes import (
    EpisodeSubtasks,
    group_by_episode,
    iter_task_episodes,
    load_task_episodes,
)
from subtask_checker.data.prepare import to_task_example
from subtask_checker.data.source import SourceSubtaskRow


def _example(row_dict, **overrides):
    row = SourceSubtaskRow.model_validate({**row_dict, **overrides})
    return to_task_example(row, "f", overrides.get("step_index", 0))


@pytest.fixture
def examples(row_dict):
    """Two episodes: episode_000017 with steps 0,1 (given out of order) and
    episode_000002 with step 0."""
    return [
        _example(row_dict, step_index=1, start_time=10.0, end_time=20.0),
        _example(row_dict, step_index=0),
        _example(row_dict, episode_id="episode_000002", episode_index=2),
    ]


def test_groups_and_orders_by_episode_then_step(examples):
    episodes = group_by_episode(examples)
    assert [e.episode_id for e in episodes] == ["episode_000002", "episode_000017"]
    ep17 = episodes[1]
    assert [ex.step_index for ex in ep17.subtasks] == [0, 1]
    assert ep17.n_subtasks == 2


def test_identity_hoisted_from_members(examples):
    ep = group_by_episode(examples)[1]
    first = ep.subtasks[0]
    assert (ep.dataset, ep.task, ep.episode_id, ep.episode_index) == (
        first.dataset, first.task, first.episode_id, first.episode_index,
    )
    assert ep.instruction == first.instruction
    assert ep.video == first.video


def test_mixed_goal_rejected(row_dict):
    a = _example(row_dict, step_index=0)
    b = _example(row_dict, step_index=1, goal="a different instruction")
    with pytest.raises(ValueError, match="goal instruction"):
        group_by_episode([a, b])


def test_mixed_video_window_rejected(row_dict):
    a = _example(row_dict, step_index=0)
    b = _example(row_dict, step_index=1, source_from_timestamp=0.0)
    with pytest.raises(ValueError, match="video window"):
        group_by_episode([a, b])


def test_duplicate_step_index_rejected(row_dict):
    a = _example(row_dict, step_index=0)
    b = _example(row_dict, step_index=0, start_time=1.0, end_time=2.0)
    with pytest.raises(ValueError, match="unique and ascending"):
        EpisodeSubtasks(
            dataset=a.dataset, task=a.task, episode_id=a.episode_id,
            episode_index=a.episode_index, instruction=a.instruction,
            video=a.video, subtasks=[a, b],
        )


def test_empty_episode_rejected(row_dict):
    a = _example(row_dict)
    with pytest.raises(ValueError):
        EpisodeSubtasks(
            dataset=a.dataset, task=a.task, episode_id=a.episode_id,
            episode_index=a.episode_index, instruction=a.instruction,
            video=a.video, subtasks=[],
        )


def test_load_task_episodes(jsonl_file):
    episodes, report = load_task_episodes(jsonl_file)
    assert report == {"rows": 2, "invalid_rows": 0, "episodes": 1, "invalid_episodes": []}
    assert len(episodes) == 1
    assert [ex.step_index for ex in episodes[0].subtasks] == [0, 1]
    # provenance falls back to the literal path for files outside the data root
    assert episodes[0].subtasks[0].provenance.source_file_server_path == str(jsonl_file)


def test_load_task_episodes_skips_and_reports(tmp_path, row_dict):
    lines = [
        json.dumps(row_dict),
        json.dumps({**row_dict, "step_index": 1, "status": "failed"}),  # invalid row
        # episode whose rows disagree about the goal -> whole episode dropped
        json.dumps({**row_dict, "episode_id": "episode_000099", "episode_index": 99}),
        json.dumps({**row_dict, "episode_id": "episode_000099", "episode_index": 99,
                    "step_index": 1, "goal": "something else"}),
    ]
    p = tmp_path / "open_vocab_subtasks.jsonl"
    p.write_text("\n".join(lines) + "\n")
    episodes, report = load_task_episodes(p)
    assert report["rows"] == 4
    assert report["invalid_rows"] == 1
    assert report["episodes"] == 1
    assert report["invalid_episodes"] == ["episode_000099"]
    assert episodes[0].episode_id == "episode_000017"


def test_iter_task_episodes_walks_data_root(tmp_path, row_dict):
    for task in ("b_task", "a_task"):
        d = tmp_path / task / config.SUBTASKS_JSONL_RELPATH.parent
        d.mkdir(parents=True)
        (d / config.SUBTASKS_JSONL_RELPATH.name).write_text(
            json.dumps({**row_dict, "task": task}) + "\n"
        )
    (tmp_path / "TEST-ignored" / config.SUBTASKS_JSONL_RELPATH.parent).mkdir(parents=True)

    out = list(iter_task_episodes(data_root=tmp_path))
    assert [t for t, _, _ in out] == ["a_task", "b_task"]
    task, episodes, report = out[0]
    assert episodes[0].task == "a_task"
    assert report["episodes"] == 1
    # provenance recorded in the server's own path form when under the data root
    assert episodes[0].subtasks[0].provenance.source_file_server_path == (
        f"{config.SERVER_DATA_PREFIX}/a_task/{config.SUBTASKS_JSONL_RELPATH}"
    )

    only = list(iter_task_episodes(data_root=tmp_path, tasks=["b_task"]))
    assert [t for t, _, _ in only] == ["b_task"]
