"""Configurable multi-grid representation of one example's video window.

Design constraint this module exists for: the eventual evaluator is a ~4B VLM
with limited practical context, so the goal is neither image quality nor frame
count but USEFUL TEMPORAL INFORMATION PER UNIT OF VISUAL CONTEXT. Both extremes
lose — tiles too small to show a grasp are wasted context, and near-duplicate
high-resolution frames are too. Where the useful middle sits is an empirical
question, so this module makes the coupled knobs explicit and cheap to vary:

    GridConfig: rows x columns (frames per grid), sample_fps, tile_width
    derived:    chunk duration, output dimensions, per-frame dimensions,
                estimated visual-token cost

and records everything needed to compare candidate configurations honestly
(``scripts/compare_grid_configs.py`` renders several for the same example).
No configuration here is claimed optimal; ``CANDIDATE_CONFIGS`` are starting
points chosen to hold per-grid token cost roughly constant while varying frame
density, so the density/legibility trade-off is isolated.

Sampling is uniform at ``sample_fps`` across the window — the deliberate
baseline. Boundary-aware or redundancy-aware sampling would slot in as an
alternative planner producing the same ``GridSetPlan``; do not add one until an
experiment justifies it. Badge times count from the window start on ONE shared
clock across all grids of a set, so boundary reasoning never needs per-grid
arithmetic.
"""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from subtask_checker.data.schema import TaskExample
from subtask_checker.video.frames import (
    CONTEXT_PAD_SEC,
    ExtractedFrame,
    VideoProbe,
    compute_window,
    extract_frame,
    probe_video,
)
from subtask_checker.video.grid import JPEG_QUALITY, GridMeta, build_grid

# Qwen2/3-VL tokenises images in 14 px ViT patches merged 2x2, i.e. one visual
# token per 28x28 px region (before any model-side rescaling). An estimate for
# comparing configs, not an exact count for any specific serving setup.
VISUAL_TOKEN_PATCH_PX = 28

# A config whose frame count explodes (high fps over a long window) should fail
# loudly at planning time, not silently truncate coverage.
MAX_TOTAL_FRAMES = 400

Scope = Literal["segment", "episode"]


class GridConfig(BaseModel):
    """One candidate frame/grid density. Everything downstream is derived."""

    model_config = ConfigDict(extra="forbid")

    name: str
    rows: int = Field(ge=1)
    columns: int = Field(ge=1)
    sample_fps: float = Field(gt=0)
    tile_width: int = Field(ge=64)
    jpeg_quality: int = JPEG_QUALITY
    badge_text_size: int | None = None  # None -> scaled from tile_width
    context_pad_sec: float = CONTEXT_PAD_SEC

    @property
    def frames_per_grid(self) -> int:
        return self.rows * self.columns

    @property
    def sample_interval_sec(self) -> float:
        return 1.0 / self.sample_fps

    @property
    def chunk_duration_sec(self) -> float:
        """Seconds of footage one full grid represents."""
        return self.frames_per_grid * self.sample_interval_sec


# Starting candidates for the density experiment. Grid pixel area (and therefore
# estimated token cost per grid) is held near-constant at ~1.0-1.2 MP while
# frames per grid go 12 -> 25 -> 36 -> 64, so what varies is temporal coverage
# versus per-frame legibility. Source is 640x480, so tile widths are downscale
# factors 0.525 / 0.4 / 0.325 / 0.25 of the source.
CANDIDATE_CONFIGS = [
    # the geometry proven in the previous evaluation project, as the baseline
    GridConfig(name="a_12f_tile336_1fps", rows=3, columns=4, sample_fps=1.0, tile_width=336),
    GridConfig(name="b_25f_tile256_2fps", rows=5, columns=5, sample_fps=2.0, tile_width=256),
    GridConfig(name="c_36f_tile208_3fps", rows=6, columns=6, sample_fps=3.0, tile_width=208),
    GridConfig(name="d_64f_tile160_4fps", rows=8, columns=8, sample_fps=4.0, tile_width=160),
]


class SingleGridPlan(BaseModel):
    grid_index: int
    sample_times_chunk_sec: list[float]
    badge_times_sec: list[float]


class GridSetPlan(BaseModel):
    """Uniform sampling of one window, chunked into grids of rows x columns.

    The last grid may be partially filled rather than stretching the interval —
    a partially filled grid is visible and honest; a silently changed sampling
    rate is not.
    """

    config: GridConfig
    scope: Scope
    window_start_chunk_sec: float
    window_end_chunk_sec: float
    segment_badge_start_sec: float
    segment_badge_end_sec: float
    grids: list[SingleGridPlan]

    @property
    def n_frames_planned(self) -> int:
        return sum(len(g.sample_times_chunk_sec) for g in self.grids)


def plan_grid_set(
    example: TaskExample,
    config: GridConfig,
    scope: Scope = "segment",
    max_total_frames: int = MAX_TOTAL_FRAMES,
) -> GridSetPlan:
    window_start, window_end = compute_window(
        example,
        context_pad_sec=config.context_pad_sec,
        full_episode=(scope == "episode"),
    )
    interval = config.sample_interval_sec
    n_frames = math.floor((window_end - window_start) / interval + 1e-6) + 1
    if n_frames > max_total_frames:
        raise ValueError(
            f"config {config.name!r} plans {n_frames} frames over a "
            f"{window_end - window_start:.1f}s window (max {max_total_frames}); "
            "lower sample_fps or narrow the window"
        )

    badge_times = [round(k * interval, 3) for k in range(n_frames)]
    chunk_times = [round(window_start + t, 3) for t in badge_times]

    per_grid = config.frames_per_grid
    grids = [
        SingleGridPlan(
            grid_index=i,
            sample_times_chunk_sec=chunk_times[start : start + per_grid],
            badge_times_sec=badge_times[start : start + per_grid],
        )
        for i, start in enumerate(range(0, n_frames, per_grid))
    ]

    seg_start, seg_end = example.segment_chunk_window()
    return GridSetPlan(
        config=config,
        scope=scope,
        window_start_chunk_sec=round(window_start, 3),
        window_end_chunk_sec=round(window_end, 3),
        segment_badge_start_sec=round(seg_start - window_start, 3) + 0.0,
        segment_badge_end_sec=round(seg_end - window_start, 3) + 0.0,
        grids=grids,
    )


class BuiltGrid(BaseModel):
    grid_index: int
    jpeg: bytes
    meta: GridMeta


class GridSetMetrics(BaseModel):
    """Everything needed to weigh a config's temporal coverage against its
    visual detail and context cost — recorded, not judged."""

    config_name: str
    scope: Scope
    # source facts (from ffprobe; None when the stream could not be probed)
    source_width: int | None
    source_height: int | None
    source_fps: float | None
    source_codec: str | None
    episode_duration_sec: float | None
    # temporal coverage
    window_duration_sec: float
    sample_fps: float
    sample_interval_sec: float
    chunk_duration_sec: float
    n_frames_planned: int
    n_frames_extracted: int
    # geometry
    n_grids: int
    rows: int
    columns: int
    frames_per_grid: int
    tile_width: int
    tile_height: int
    full_grid_width: int
    full_grid_height: int
    downscale_vs_source: float | None
    # context cost
    total_megapixels: float
    est_visual_tokens: int
    est_visual_tokens_per_window_sec: float


def estimate_visual_tokens(width: int, height: int) -> int:
    return math.ceil(width / VISUAL_TOKEN_PATCH_PX) * math.ceil(height / VISUAL_TOKEN_PATCH_PX)


def build_grid_set(
    video_path,
    example: TaskExample,
    plan: GridSetPlan,
) -> tuple[list[BuiltGrid], GridSetMetrics]:
    """Extract and tile every grid in the plan. Unreadable timestamps are dropped
    (same policy as extract_frames); a grid with no readable frames is dropped;
    raises ValueError if nothing at all could be extracted."""
    config = plan.config
    built: list[BuiltGrid] = []
    n_extracted = 0
    for grid_plan in plan.grids:
        frames = []
        for chunk_t, badge_t in zip(
            grid_plan.sample_times_chunk_sec, grid_plan.badge_times_sec
        ):
            jpeg = extract_frame(video_path, chunk_t)
            if jpeg is not None:
                frames.append(
                    ExtractedFrame(badge_time_sec=badge_t, chunk_time_sec=chunk_t, jpeg=jpeg)
                )
        n_extracted += len(frames)
        if not frames:
            continue
        grid_jpeg, meta = build_grid(
            frames,
            columns=config.columns,
            tile_width=config.tile_width,
            quality=config.jpeg_quality,
            badge_text_size=config.badge_text_size,
        )
        built.append(BuiltGrid(grid_index=grid_plan.grid_index, jpeg=grid_jpeg, meta=meta))
    if not built:
        raise ValueError(f"no frames could be extracted from {video_path}")

    metrics = compute_metrics(example, plan, built, probe_video(video_path), n_extracted)
    return built, metrics


def compute_metrics(
    example: TaskExample,
    plan: GridSetPlan,
    built: list[BuiltGrid],
    probe: VideoProbe | None,
    n_extracted: int,
) -> GridSetMetrics:
    config = plan.config
    tile_height = built[0].meta.tile_height
    window_duration = plan.window_end_chunk_sec - plan.window_start_chunk_sec
    total_pixels = sum(g.meta.width * g.meta.height for g in built)
    est_tokens = sum(estimate_visual_tokens(g.meta.width, g.meta.height) for g in built)
    return GridSetMetrics(
        config_name=config.name,
        scope=plan.scope,
        source_width=probe.width if probe else None,
        source_height=probe.height if probe else None,
        source_fps=probe.fps if probe else None,
        source_codec=probe.codec if probe else None,
        episode_duration_sec=example.video.episode_duration_sec,
        window_duration_sec=round(window_duration, 3),
        sample_fps=config.sample_fps,
        sample_interval_sec=round(config.sample_interval_sec, 4),
        chunk_duration_sec=round(config.chunk_duration_sec, 3),
        n_frames_planned=plan.n_frames_planned,
        n_frames_extracted=n_extracted,
        n_grids=len(built),
        rows=config.rows,
        columns=config.columns,
        frames_per_grid=config.frames_per_grid,
        tile_width=config.tile_width,
        tile_height=tile_height,
        full_grid_width=config.columns * config.tile_width,
        full_grid_height=config.rows * tile_height,
        downscale_vs_source=(
            round(config.tile_width / probe.width, 4) if probe else None
        ),
        total_megapixels=round(total_pixels / 1e6, 3),
        est_visual_tokens=est_tokens,
        est_visual_tokens_per_window_sec=round(est_tokens / max(window_duration, 1e-6), 1),
    )
