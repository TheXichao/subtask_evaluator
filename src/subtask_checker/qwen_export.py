"""EvaluationInput → the official Qwen fine-tuning annotation format.

Target format (verified 2026-08-12 against ``QwenLM/Qwen3-VL/qwen-vl-finetune``,
byte-identical to Qwen2.5-VL's; details and sources in docs/qwen_dataset.md):

    {"image": ["<relpath>", ...],
     "conversations": [{"from": "human", "value": "<image>\\n<image>\\n..."},
                       {"from": "gpt",   "value": "..."}]}

Rules the trainer enforces that this module therefore validates up front:
exactly one ``<image>`` tag per listed image, consumed left-to-right in list
order; tags never in the assistant turn; no system role exists in the format,
so the evaluation instruction must live inside the human turn. Media paths are
resolved against the dataset's registered ``data_path`` root, so every image
path here is relative to this export's ``media/`` directory and the export
directory is self-contained.

Supervision boundary (hard rule: no human labels at this stage, and generated
subtasks are never ground truth): the assistant turn requires an explicit
:class:`QwenTarget` carrying its own ``source`` tag, recorded per record in the
manifest — supervision sources can never be silently mixed. Until real targets
exist, :data:`PLACEHOLDER_TARGET` makes an input-only export loud and
unmistakable; a dataset carrying it is for inspection and pipeline validation,
not for training.

The same records are simultaneously valid LLaMA-Factory ShareGPT data —
LLaMA-Factory's default tags are exactly ``from``/``value``/``human``/``gpt``
and its ``images`` column is remappable to our ``image`` key — so one export
serves both trainers; only the registration differs (:func:`registration_snippet`
vs :func:`llama_factory_snippet`, verified facts in docs/qwen_dataset.md).

This module renders no second grid algorithm and calls no model: grids come
from :func:`subtask_checker.eval_input.prepare_evaluation_input` unchanged.
"""

import json
import re
from pathlib import Path, PurePosixPath
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from subtask_checker import config
from subtask_checker.data.schema import TaskExample
from subtask_checker.eval_input import (
    DEFAULT_EVALUATION_INSTRUCTION,
    DEFAULT_GRID_CONFIG,
    EvaluationInput,
    prepare_evaluation_input,
    write_evaluation_input,
)
from subtask_checker.video.gridset import GridConfig, GridSetMetrics, Scope

IMAGE_TAG = "<image>"
VIDEO_TAG = "<video>"

ANNOTATIONS_FILENAME = "annotations.json"
MANIFEST_FILENAME = "manifest.json"
MEDIA_DIRNAME = "media"
EVALUATION_INPUT_FILENAME = "evaluation_input.json"

# Dataset names become directory names and registry keys.
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

# The one prompt shape of this export: image tags first (list order == tag
# order is a trainer requirement), then the evaluation instruction, then the
# structured facts of THIS subtask. Times are given on the burned-in badge
# clock — the only clock the model can see (the official framework carries no
# timestamp channel).
HUMAN_TURN_TEMPLATE = (
    "{image_tags}\n"
    "{evaluation_instruction}\n"
    "\n"
    "Episode goal: {goal}\n"
    "Annotated subtask: {description}\n"
    "Claimed segment: {badge_start:.1f}s to {badge_end:.1f}s on the clock "
    "burned into the frames."
)


class QwenTarget(BaseModel):
    """One assistant turn plus where it came from. ``source`` is mandatory so
    an annotation file can always be traced to its supervision source."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    source: str = Field(min_length=1)

    @field_validator("source")
    @classmethod
    def _never_ground_truth(cls, v: str) -> str:
        if v == "ground_truth":
            raise ValueError(
                "'ground_truth' is banned as a supervision source name — "
                "generated subtasks are the thing being judged (CLAUDE.md rule 4); "
                "name the actual source, e.g. 'human_review_v1'"
            )
        return v


PLACEHOLDER_TARGET = QwenTarget(
    text=(
        "[NO TARGET YET] No supervision target has been constructed for this "
        "example. This record is input-side only; a dataset containing this "
        "text must not be trained on."
    ),
    source="placeholder_pending_labels",
)


class QwenTurn(BaseModel):
    """One conversation turn, with the trainer's own key names on the wire
    (``from``/``value``; serialise with ``by_alias=True``)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    role: Literal["human", "gpt"] = Field(alias="from")
    value: str = Field(min_length=1)


class QwenSample(BaseModel):
    """One annotation record, restricted to exactly the official keys.

    Everything the trainer would reject at training time is rejected here at
    export time instead: tag/image count mismatch, tags in assistant turns,
    non-alternating roles. Sidecar metadata (provenance, configs) deliberately
    lives in the manifest, not in the record.
    """

    model_config = ConfigDict(extra="forbid")

    image: list[str] = Field(min_length=1)
    conversations: list[QwenTurn] = Field(min_length=2)

    @field_validator("image")
    @classmethod
    def _relative_paths(cls, v: list[str]) -> list[str]:
        for p in v:
            if not p.strip():
                raise ValueError("empty image path")
            if PurePosixPath(p).is_absolute() or Path(p).is_absolute():
                raise ValueError(
                    f"image path must be relative to the dataset media root, got {p!r}"
                )
        return v

    @model_validator(mode="after")
    def _trainer_invariants(self) -> "QwenSample":
        roles = [t.role for t in self.conversations]
        expected = ["human", "gpt"] * (len(roles) // 2)
        if roles != expected:
            raise ValueError(f"turns must alternate human/gpt starting with human, got {roles}")
        n_tags = sum(t.value.count(IMAGE_TAG) for t in self.conversations if t.role == "human")
        if n_tags != len(self.image):
            raise ValueError(
                f"{n_tags} {IMAGE_TAG} tag(s) for {len(self.image)} image(s) — "
                "the trainer requires exactly one tag per image"
            )
        for t in self.conversations:
            if t.role == "gpt" and (IMAGE_TAG in t.value or VIDEO_TAG in t.value):
                raise ValueError("media tags must not appear in the assistant turn")
            if VIDEO_TAG in t.value:
                raise ValueError(
                    f"{VIDEO_TAG} tag without a video entry would fail in the trainer"
                )
        return self

    def to_annotation(self) -> dict:
        """The dict exactly as it goes into annotations.json."""
        return self.model_dump(by_alias=True)


def build_human_turn(eval_input: EvaluationInput) -> str:
    """Render the user turn: one image tag per grid, instruction, subtask facts."""
    return HUMAN_TURN_TEMPLATE.format(
        image_tags="\n".join(IMAGE_TAG for _ in eval_input.visual_inputs),
        evaluation_instruction=eval_input.evaluation_instruction,
        goal=eval_input.example.instruction,
        description=eval_input.example.segment.description,
        badge_start=eval_input.plan.segment_badge_start_sec,
        badge_end=eval_input.plan.segment_badge_end_sec,
    )


def to_qwen_sample(
    eval_input: EvaluationInput, target: QwenTarget, media_subdir: str = ""
) -> QwenSample:
    """One EvaluationInput → one annotation record.

    ``media_subdir`` re-roots the grid paths (relative to the example's own
    directory) onto the dataset media root, e.g. the example's directory name
    inside ``media/``.
    """
    prefix = PurePosixPath(media_subdir) if media_subdir else None
    return QwenSample(
        image=[str(prefix / v.path) if prefix else v.path for v in eval_input.visual_inputs],
        conversations=[
            QwenTurn(role="human", value=build_human_turn(eval_input)),
            QwenTurn(role="gpt", value=target.text),
        ],
    )


class QwenRecordMeta(BaseModel):
    """Per-record sidecar: what the annotation format itself cannot carry."""

    index: int
    example_id: str
    annotation_source: str  # from TaskExample.provenance
    target_source: str
    n_images: int
    est_visual_tokens: int


class QwenDatasetManifest(BaseModel):
    """The export's own record: resolved configs and provenance, so the run
    reproduces from the repo alone and no record is untraceable."""

    model_config = ConfigDict(extra="forbid")

    name: str
    annotation_file: str = ANNOTATIONS_FILENAME
    media_dir: str = MEDIA_DIRNAME
    grid_config: GridConfig
    scope: Scope
    evaluation_instruction: str
    # The largest rendered grid in raw pixels: the trainer's global --max_pixels
    # must be at least this or it silently downscales the grids (docs/qwen_dataset.md).
    max_grid_pixels: int
    records: list[QwenRecordMeta]
    skipped_missing_video: list[str] = []
    skipped_render_failed: list[str] = []

    @property
    def target_sources(self) -> list[str]:
        return sorted({r.target_source for r in self.records})


def export_qwen_dataset(
    examples: Sequence[TaskExample],
    out_dir: str | Path,
    name: str,
    targets: Mapping[str, QwenTarget] | None = None,
    grid_config: GridConfig = DEFAULT_GRID_CONFIG,
    scope: Scope = "segment",
    instruction: str = DEFAULT_EVALUATION_INSTRUCTION,
) -> QwenDatasetManifest:
    """Render grids for every example and write a self-contained dataset dir:

        <out_dir>/
        ├── annotations.json    official format; image paths relative to media/
        ├── media/<example>/... grid images + the inspectable EvaluationInput
        └── manifest.json       resolved configs + per-record provenance

    ``targets`` maps example_id → :class:`QwenTarget`; when given it must cover
    every exported example (partial coverage would silently mix supervision
    sources). When ``None``, every record carries :data:`PLACEHOLDER_TARGET`.
    Examples whose video is unreachable or yields no frames are skipped and
    reported in the manifest, never silently dropped.
    """
    if not _NAME_RE.match(name):
        raise ValueError(f"dataset name {name!r} must match {_NAME_RE.pattern}")
    ids = [ex.example_id for ex in examples]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate example_ids in export: {dupes}")
    if targets is not None:
        missing = [i for i in ids if i not in targets]
        if missing:
            raise ValueError(
                f"targets missing for {len(missing)} example(s), e.g. {missing[:3]} — "
                "provide a target for every example or none"
            )

    out = config.require_writable(Path(out_dir))
    media_root = out / MEDIA_DIRNAME
    media_root.mkdir(parents=True, exist_ok=True)

    samples: list[QwenSample] = []
    metas: list[QwenRecordMeta] = []
    manifest = QwenDatasetManifest(
        name=name,
        grid_config=grid_config,
        scope=scope,
        evaluation_instruction=instruction,
        max_grid_pixels=0,
        records=[],
    )
    for ex in examples:
        video_path = ex.video.local_path()
        if not video_path.is_file():
            manifest.skipped_missing_video.append(ex.example_id)
            continue
        example_dirname = ex.example_id.replace("/", "__")
        try:
            eval_input, metrics = prepare_evaluation_input(
                ex, video_path, media_root / example_dirname,
                grid_config=grid_config, scope=scope, instruction=instruction,
            )
        except ValueError:
            manifest.skipped_render_failed.append(ex.example_id)
            continue
        write_evaluation_input(
            eval_input, media_root / example_dirname / EVALUATION_INPUT_FILENAME
        )
        target = targets[ex.example_id] if targets is not None else PLACEHOLDER_TARGET
        samples.append(to_qwen_sample(eval_input, target, media_subdir=example_dirname))
        metas.append(
            QwenRecordMeta(
                index=len(metas),
                example_id=ex.example_id,
                annotation_source=ex.provenance.annotation_source,
                target_source=target.source,
                n_images=len(eval_input.visual_inputs),
                est_visual_tokens=metrics.est_visual_tokens,
            )
        )
        manifest.max_grid_pixels = max(
            manifest.max_grid_pixels, _max_grid_pixels(metrics)
        )

    if not samples:
        raise ValueError(f"nothing to export: all {len(ids)} example(s) were skipped")
    manifest.records = metas

    (out / ANNOTATIONS_FILENAME).write_text(
        json.dumps([s.to_annotation() for s in samples], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out / MANIFEST_FILENAME).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _max_grid_pixels(metrics: GridSetMetrics) -> int:
    return metrics.full_grid_width * metrics.full_grid_height


def read_annotations(dataset_dir: str | Path) -> list[QwenSample]:
    """Load and re-validate an export's annotation file (the inspection path)."""
    raw = json.loads(
        (Path(dataset_dir) / ANNOTATIONS_FILENAME).read_text(encoding="utf-8")
    )
    return [QwenSample.model_validate(r) for r in raw]


def missing_media_files(dataset_dir: str | Path) -> list[Path]:
    """Referenced images that do not exist under the export's media root."""
    root = Path(dataset_dir) / MEDIA_DIRNAME
    return [
        root / p
        for s in read_annotations(dataset_dir)
        for p in s.image
        if not (root / p).is_file()
    ]


def registration_snippet(name: str, dataset_dir: str | Path) -> str:
    """The entry to add to ``qwen-vl-finetune/qwenvl/data/__init__.py`` — the
    official framework registers datasets in a hard-coded dict there."""
    d = Path(dataset_dir).resolve()
    var = re.sub(r"[^A-Za-z0-9]", "_", name).upper()
    return (
        f"{var} = {{\n"
        f'    "annotation_path": "{d / ANNOTATIONS_FILENAME}",\n'
        f'    "data_path": "{d / MEDIA_DIRNAME}",\n'
        f"}}\n"
        f'data_dict["{name}"] = {var}'
    )


def llama_factory_snippet(name: str, dataset_dir: str | Path) -> str:
    """The entry to merge into LLaMA-Factory's ``data/dataset_info.json``.

    The annotation file is used unchanged: LLaMA-Factory's ShareGPT defaults
    already match our record keys, so only the ``image`` column needs
    remapping. Media paths stay relative to ``media/``, which is NOT a
    dataset_info concept — the training YAML must set ``media_dir`` to this
    export's ``media/`` directory (see docs/qwen_dataset.md).
    """
    d = Path(dataset_dir).resolve()
    return json.dumps(
        {
            name: {
                "file_name": str(d / ANNOTATIONS_FILENAME),
                "formatting": "sharegpt",
                "columns": {"messages": "conversations", "images": "image"},
            }
        },
        indent=2,
    )
