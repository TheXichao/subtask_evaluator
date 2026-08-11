"""Shared project/environment configuration: the canonical paths, nothing else.

One rule decides what belongs here: only settings that identify shared
infrastructure — where the repo, the read-only external mounts, and the
generated outputs live. Everything else has a different home:

  * component knobs (grid geometry, sampling) → the component's own config
    models, e.g. :mod:`subtask_checker.video.gridset`;
  * experiment variations → ``experiments/*.json`` via
    :mod:`subtask_checker.experiments`;
  * facts about the team's data layout (server path prefix, JSONL location) →
    domain constants below, in code — they describe the dataset, not our
    environment, and no experiment may "configure" them.

The dataset lives on the GPU server and is sshfs-mounted locally. Team JSON
stores server-absolute paths (``/mnt/raid0/...``) which must be translated to
the local mount before use. Nothing in this project ever writes to the mounts
or to the old evaluation project; :func:`require_writable` enforces that at
the write sites.
"""

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict

# --- facts about the team's data layout: domain constants, NOT configuration ---

# Prefix that server-side JSON uses for the dataset tree.
SERVER_DATA_PREFIX = "/mnt/raid0/supeng_g1_data/410_g1_ir"

# Where the pre-review subtasker output lives inside each task directory.
SUBTASKS_JSONL_RELPATH = Path("meta/motion_desc_pipeline/open_vocab_subtasks.jsonl")

# Task-dir names that are not LeRobot task datasets (test copies, raw teleop zarr,
# v2.1 conversion smoke tests).
EXCLUDED_TASK_PREFIXES = ("TEST-", "Motor_STAGE")

# Repo root. Valid because this project always runs from its own checkout (uv).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProjectPaths(BaseModel):
    """Canonical locations, frozen so the shared default cannot drift at runtime.

    Tests and special setups construct their own instance (or set the
    ``SUBTASK_CHECKER_*`` environment variables) instead of mutating ``PATHS``.
    Existence is deliberately not validated here — the mount may be down, and
    callers decide whether a missing file is fatal.
    """

    model_config = ConfigDict(frozen=True)

    project_root: Path
    # sshfs mount of unitree@10.0.8.204:/mnt/raid0/supeng_g1_data/410_g1_ir — READ-ONLY
    data_root: Path
    # sshfs mount of the team's subtasker repo (Unitree_Robo_Describer) — READ-ONLY
    team_repo_root: Path
    # previous evaluation project: reference implementations only — never modified
    evaluation_reference_root: Path
    # the only tree processing code may write into (gitignored, reproducible)
    output_root: Path

    @classmethod
    def from_env(cls) -> "ProjectPaths":
        def env_path(var: str, default: Path) -> Path:
            return Path(os.environ.get(var, str(default))).expanduser()

        home = Path.home()
        return cls(
            project_root=_PROJECT_ROOT,
            data_root=env_path("SUBTASK_CHECKER_DATA_ROOT", home / "remote" / "410_g1_ir"),
            team_repo_root=env_path(
                "SUBTASK_CHECKER_TEAM_REPO_ROOT", home / "remote" / "Unitree_Robo_Describer"
            ),
            evaluation_reference_root=env_path(
                "SUBTASK_CHECKER_EVALUATION_ROOT", home / "Dev" / "evaluation"
            ),
            output_root=env_path("SUBTASK_CHECKER_OUTPUT_ROOT", _PROJECT_ROOT / "data"),
        )

    @property
    def read_only_roots(self) -> tuple[Path, ...]:
        return (self.data_root, self.team_repo_root, self.evaluation_reference_root)

    @property
    def samples_dir(self) -> Path:
        return self.output_root / "samples"

    @property
    def dev_sample_jsonl(self) -> Path:
        """The canonical dev sample: written by build_sample.py, read by the
        example/experiment scripts — shared, so named exactly once."""
        return self.samples_dir / "dev_sample.jsonl"


PATHS = ProjectPaths.from_env()


def resolve_server_path(path: str | Path, data_root: Path | None = None) -> Path:
    """Translate a server-absolute path from team JSON to the local mount.

    Returns the path unchanged if it does not start with the known server prefix.
    Existence is not checked here; callers decide whether a missing file is fatal.
    """
    root = data_root if data_root is not None else PATHS.data_root
    s = str(path)
    if s.startswith(SERVER_DATA_PREFIX):
        return root / s[len(SERVER_DATA_PREFIX) :].lstrip("/")
    return Path(s)


def require_writable(path: str | Path, paths: ProjectPaths | None = None) -> Path:
    """Refuse a write target under any read-only external root (hard rule #1).

    Called at the write sites for generated artifacts; returns the path
    unchanged so it can wrap the target inline.
    """
    resolved = Path(path).expanduser().resolve()
    for root in (paths if paths is not None else PATHS).read_only_roots:
        if resolved.is_relative_to(root.expanduser().resolve()):
            raise ValueError(f"refusing to write under read-only root {root}: {path}")
    return Path(path)
