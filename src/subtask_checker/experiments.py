"""Experiment configuration: named, validated override files.

An experiment varies component defaults without touching source code. The
composition is deliberately plain:

    component defaults (Pydantic field defaults on the component's config
    models, e.g. video/gridset.py::GridConfig)
        + an experiments/*.json file specifying only what it varies
        → validated typed configs handed to the component

Pydantic fills unspecified fields from the component defaults and rejects
unknown fields, so an experiment cannot silently configure something that
does not exist. Experiment files are source-controlled (``experiments/`` at
the repo root) and generated artifacts record the experiment name plus the
fully resolved configs, so any run is reproducible from the repo alone.

Today an experiment only varies the video grid representation — the component
under development. When another component grows experiment-varied settings,
its config model becomes one more optional field on :class:`ExperimentConfig`;
that is the entire extension mechanism. No registry, no discovery.
"""

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from subtask_checker.video.gridset import CANDIDATE_CONFIGS, GridConfig

# Experiment names become directory names under the outputs tree.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ExperimentConfig(BaseModel):
    """One named experiment: which grid configurations to render and why."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    grids: list[GridConfig] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _name_is_path_safe(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError(
                f"experiment name {v!r} must be filesystem-safe: letters, digits, '._-'"
            )
        return v

    @model_validator(mode="after")
    def _unique_grid_names(self) -> "ExperimentConfig":
        names = [g.name for g in self.grids]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate grid config names {dupes}: outputs are keyed by name")
        return self


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Parse and validate one experiment file; raises on anything malformed."""
    return ExperimentConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def default_experiment() -> ExperimentConfig:
    """The built-in density candidates, as the experiment run when none is given."""
    return ExperimentConfig(
        name="candidates",
        description="built-in density candidates from video/gridset.py",
        grids=list(CANDIDATE_CONFIGS),
    )
