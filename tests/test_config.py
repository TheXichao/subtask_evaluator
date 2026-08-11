from pathlib import Path

import pytest
from pydantic import ValidationError

from subtask_checker import config
from subtask_checker.config import ProjectPaths, require_writable

ENV_VARS = (
    "SUBTASK_CHECKER_DATA_ROOT",
    "SUBTASK_CHECKER_TEAM_REPO_ROOT",
    "SUBTASK_CHECKER_EVALUATION_ROOT",
    "SUBTASK_CHECKER_OUTPUT_ROOT",
)


@pytest.fixture
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_default_paths_point_at_this_repo(clean_env):
    paths = ProjectPaths.from_env()
    assert (paths.project_root / "pyproject.toml").is_file()
    assert paths.output_root == paths.project_root / "data"
    assert paths.dev_sample_jsonl == paths.output_root / "samples" / "dev_sample.jsonl"
    assert paths.data_root == Path.home() / "remote" / "410_g1_ir"


def test_env_overrides(clean_env, tmp_path):
    clean_env.setenv("SUBTASK_CHECKER_DATA_ROOT", str(tmp_path / "mount"))
    clean_env.setenv("SUBTASK_CHECKER_OUTPUT_ROOT", str(tmp_path / "out"))
    paths = ProjectPaths.from_env()
    assert paths.data_root == tmp_path / "mount"
    assert paths.output_root == tmp_path / "out"
    # untouched fields keep their defaults
    assert paths.team_repo_root == Path.home() / "remote" / "Unitree_Robo_Describer"


def test_paths_are_frozen():
    with pytest.raises(ValidationError):
        config.PATHS.data_root = Path("/elsewhere")


def test_external_roots_are_registered_read_only():
    assert config.PATHS.data_root in config.PATHS.read_only_roots
    assert config.PATHS.team_repo_root in config.PATHS.read_only_roots
    assert config.PATHS.evaluation_reference_root in config.PATHS.read_only_roots
    assert config.PATHS.output_root not in config.PATHS.read_only_roots


@pytest.fixture
def fake_paths(tmp_path):
    return ProjectPaths(
        project_root=tmp_path,
        data_root=tmp_path / "mount",
        team_repo_root=tmp_path / "team",
        evaluation_reference_root=tmp_path / "eval",
        output_root=tmp_path / "data",
    )


def test_require_writable_rejects_every_read_only_root(fake_paths):
    for root in fake_paths.read_only_roots:
        with pytest.raises(ValueError, match="read-only"):
            require_writable(root / "sub" / "artifact.json", paths=fake_paths)


def test_require_writable_allows_output_root(fake_paths):
    target = fake_paths.output_root / "samples" / "x.jsonl"
    assert require_writable(target, paths=fake_paths) == target


def test_resolve_server_path_uses_canonical_default():
    local = config.resolve_server_path(config.SERVER_DATA_PREFIX + "/task/a.mp4")
    assert local == config.PATHS.data_root / "task" / "a.mp4"
