import json

import pytest
from pydantic import ValidationError

from subtask_checker.experiments import (
    ExperimentConfig,
    default_experiment,
    load_experiment,
)
from subtask_checker.video.gridset import CANDIDATE_CONFIGS

MINIMAL_GRID = {"name": "g1", "rows": 5, "columns": 5, "sample_fps": 2.0, "tile_width": 256}


def experiment_dict(**overrides):
    return {"name": "exp1", "description": "test", "grids": [dict(MINIMAL_GRID)], **overrides}


@pytest.fixture
def experiment_file(tmp_path):
    def write(payload):
        p = tmp_path / "exp.json"
        p.write_text(json.dumps(payload))
        return p
    return write


def test_valid_experiment_loads_typed(experiment_file):
    exp = load_experiment(experiment_file(experiment_dict()))
    assert exp.name == "exp1"
    grid = exp.grids[0]
    assert (grid.rows, grid.columns, grid.sample_fps, grid.tile_width) == (5, 5, 2.0, 256)
    assert grid.chunk_duration_sec == pytest.approx(12.5)


def test_component_defaults_fill_unspecified_fields(experiment_file):
    # the override file names only what it varies; GridConfig defaults do the rest
    exp = load_experiment(experiment_file(experiment_dict()))
    grid = exp.grids[0]
    assert grid.jpeg_quality == 95
    assert grid.context_pad_sec == 3.0
    assert grid.badge_text_size is None


def test_invalid_values_rejected(experiment_file):
    bad = experiment_dict()
    bad["grids"][0]["rows"] = 0
    with pytest.raises(ValidationError):
        load_experiment(experiment_file(bad))


def test_unknown_fields_rejected_at_both_levels(experiment_file):
    with pytest.raises(ValidationError):
        load_experiment(experiment_file(experiment_dict(learning_rate=1e-4)))
    bad = experiment_dict()
    bad["grids"][0]["fps"] = 5  # not a GridConfig field (it is sample_fps)
    with pytest.raises(ValidationError):
        load_experiment(experiment_file(bad))


def test_empty_grid_list_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig(name="e", grids=[])


def test_duplicate_grid_names_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        ExperimentConfig.model_validate(
            {"name": "e", "grids": [dict(MINIMAL_GRID), dict(MINIMAL_GRID)]}
        )


def test_experiment_name_must_be_path_safe():
    with pytest.raises(ValidationError, match="filesystem-safe"):
        ExperimentConfig.model_validate(experiment_dict(name="a/b"))


def test_default_experiment_wraps_builtin_candidates():
    exp = default_experiment()
    assert exp.name == "candidates"
    assert [g.name for g in exp.grids] == [c.name for c in CANDIDATE_CONFIGS]


def test_checked_in_experiment_files_are_valid():
    from subtask_checker.config import PATHS

    files = sorted((PATHS.project_root / "experiments").glob("*.json"))
    assert files, "expected at least one experiment file in experiments/"
    for f in files:
        exp = load_experiment(f)
        assert exp.name == f.stem, f"{f.name}: file name and experiment name must match"
