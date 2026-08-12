"""One model evaluation example's INPUT, as a validated object.

This is the contract between data preparation and the future inference layer:

    TaskExample + rendered grid images + evaluation instruction
        → EvaluationInput                       (this module)
        → [future] prompt construction → VLM    (a separate module, not built)

Everything upstream already exists — segment resolution in ``data/``, window
computation and grid rendering in ``video/`` — so this module only assembles
and records; it implements no second sampling or grid algorithm and never
talks to a model. The point of the boundary is that the exact model input can
be inspected and tested on disk before any model is involved.

The visual side is a LIST of images by design: one grid is the common case,
but a window longer than one grid's capacity already yields several (the
``GridSetPlan`` chunks one shared badge clock across grids), and density
experiments may later pair a coarse overview with a dense local view. Image
paths are stored relative to the directory holding the example's JSON, so an
example directory is self-contained and portable (docs/evaluation_input.md).
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from subtask_checker import config
from subtask_checker.data.schema import TaskExample
from subtask_checker.video.grid import GridMeta
from subtask_checker.video.gridset import (
    CANDIDATE_CONFIGS,
    BuiltGrid,
    GridConfig,
    GridSetMetrics,
    GridSetPlan,
    Scope,
    build_grid_set,
    plan_grid_set,
)

# The proven baseline geometry (12 frames / 336 px tiles); stays the default
# until the density experiments justify a change (docs/grid_experiments.md).
DEFAULT_GRID_CONFIG = CANDIDATE_CONFIGS[0]

# Development placeholder — enough to inspect complete inputs end to end. The
# real rubric and output schema are a later, separate step; do not grow this
# into one. Segment times/description are NOT interpolated here: rendering the
# structured fields into a prompt is the future inference module's job.
DEFAULT_EVALUATION_INSTRUCTION = (
    "You are shown frames from a robot manipulation episode, tiled into one or "
    "more grids in temporal order; each tile's time in seconds from the window "
    "start is burned into its top-left corner. Given the episode goal and one "
    "annotated segment (start time, end time, description), judge whether the "
    "segment is correctly labelled: does the described action occur there, and "
    "do the claimed boundaries match what is visible?"
)

GRIDS_DIRNAME = "grids"


class VisualInput(BaseModel):
    """One image handed to the model, with the metadata to read its badge clock."""

    model_config = ConfigDict(extra="forbid")

    # POSIX-style path relative to the directory holding the example's JSON —
    # never absolute, so the example directory stays self-contained.
    path: str
    grid_index: int = Field(ge=0)
    meta: GridMeta

    @field_validator("path")
    @classmethod
    def _relative_and_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("visual input path is empty")
        if Path(v).is_absolute():
            raise ValueError(
                f"visual input path must be relative to the example directory, got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _has_badges(self) -> "VisualInput":
        if not self.meta.badge_times_sec:
            raise ValueError("visual input carries no badge times")
        return self

    @property
    def badge_start_sec(self) -> float:
        return self.meta.badge_times_sec[0]

    @property
    def badge_end_sec(self) -> float:
        return self.meta.badge_times_sec[-1]


class EvaluationInput(BaseModel):
    """Everything the evaluator will receive for ONE subtask judgement.

    ``example`` carries episode identity, goal instruction, the annotated
    segment, and provenance; ``plan`` carries the sampling window, the shared
    badge clock (including where the segment sits on it), and the resolved
    :class:`GridConfig`; ``visual_inputs`` are the rendered images.
    """

    model_config = ConfigDict(extra="forbid")

    example: TaskExample
    plan: GridSetPlan
    visual_inputs: list[VisualInput] = Field(min_length=1)
    evaluation_instruction: str = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> "EvaluationInput":
        if self.plan.window_end_chunk_sec <= self.plan.window_start_chunk_sec:
            raise ValueError(
                f"degenerate window: {self.plan.window_start_chunk_sec}"
                f"-{self.plan.window_end_chunk_sec}s"
            )
        if self.plan.segment_badge_end_sec <= self.plan.segment_badge_start_sec:
            raise ValueError(
                f"degenerate segment badges: {self.plan.segment_badge_start_sec}"
                f"-{self.plan.segment_badge_end_sec}s"
            )
        indices = [v.grid_index for v in self.visual_inputs]
        if indices != sorted(set(indices)):
            raise ValueError(f"visual inputs must have unique ascending grid_index, got {indices}")
        planned = {g.grid_index for g in self.plan.grids}
        unknown = [i for i in indices if i not in planned]
        if unknown:
            raise ValueError(f"visual inputs reference grid_index {unknown} not in the plan")
        return self

    @property
    def grid_config(self) -> GridConfig:
        return self.plan.config


def build_evaluation_input(
    example: TaskExample,
    plan: GridSetPlan,
    built: list[BuiltGrid],
    grid_relpaths: list[str],
    instruction: str = DEFAULT_EVALUATION_INSTRUCTION,
) -> EvaluationInput:
    """Assemble the validated input from already-rendered grids (pure, no IO)."""
    if len(built) != len(grid_relpaths):
        raise ValueError(f"{len(built)} grids but {len(grid_relpaths)} paths")
    return EvaluationInput(
        example=example,
        plan=plan,
        visual_inputs=[
            VisualInput(path=p, grid_index=g.grid_index, meta=g.meta)
            for g, p in zip(built, grid_relpaths)
        ],
        evaluation_instruction=instruction,
    )


def prepare_evaluation_input(
    example: TaskExample,
    video_path: str | Path,
    out_dir: str | Path,
    grid_config: GridConfig = DEFAULT_GRID_CONFIG,
    scope: Scope = "segment",
    instruction: str = DEFAULT_EVALUATION_INSTRUCTION,
) -> tuple[EvaluationInput, GridSetMetrics]:
    """Render one example's grids into ``out_dir`` and assemble its input.

    ``out_dir`` is the example's own directory: grids land in
    ``<out_dir>/grids/<config_name>/`` and the returned visual paths are
    relative to ``out_dir``, so the caller should write the JSON there too
    (:func:`write_evaluation_input`). Raises if nothing could be extracted.
    """
    plan = plan_grid_set(example, grid_config, scope=scope)
    built, metrics = build_grid_set(video_path, example, plan)

    grids_dir = config.require_writable(Path(out_dir) / GRIDS_DIRNAME / grid_config.name)
    grids_dir.mkdir(parents=True, exist_ok=True)
    for stale in grids_dir.glob("grid_*.jpg"):
        stale.unlink()
    relpaths = []
    for g in built:
        name = f"grid_{g.grid_index:02d}.jpg"
        (grids_dir / name).write_bytes(g.jpeg)
        relpaths.append(f"{GRIDS_DIRNAME}/{grid_config.name}/{name}")

    return build_evaluation_input(example, plan, built, relpaths, instruction), metrics


def write_evaluation_input(eval_input: EvaluationInput, json_path: str | Path) -> None:
    out = config.require_writable(Path(json_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(eval_input.model_dump_json(indent=2), encoding="utf-8")


def read_evaluation_input(json_path: str | Path) -> EvaluationInput:
    return EvaluationInput.model_validate_json(Path(json_path).read_text(encoding="utf-8"))


def missing_visual_files(eval_input: EvaluationInput, base_dir: str | Path) -> list[Path]:
    """Referenced images that do not exist under ``base_dir`` (the JSON's directory)."""
    base = Path(base_dir)
    return [base / v.path for v in eval_input.visual_inputs if not (base / v.path).is_file()]
