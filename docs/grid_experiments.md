# Grid density experiments — visual-token efficiency

## The constraint

The eventual evaluator is a ~4B VLM with limited practical context. The grid
representation is therefore optimised for **useful temporal information per
unit of visual context** — not image quality alone, not frame count alone.
Both extremes lose: a frame too small to show a grasp is wasted context, and a
crisp frame that duplicates its neighbour is too.

Where the useful middle sits is an empirical question. The machinery for
answering it lives in `src/subtask_checker/video/gridset.py` +
`scripts/compare_grid_configs.py`.

## Knobs and derived quantities

`GridConfig` fixes the independent knobs; everything else is derived and
recorded in `GridSetMetrics`:

| knob | derived |
|---|---|
| `rows`, `columns` | frames per grid |
| `sample_fps` | sample interval, chunk duration (`frames_per_grid / fps`) |
| `tile_width` | tile height (source aspect), full-grid pixel dims, downscale vs source |
| `context_pad_sec` | window span (segment scope) |

Sampling is **uniform** at `sample_fps` over the window — the deliberate
baseline. Boundary-aware / redundancy-aware sampling would be an alternative
planner producing the same `GridSetPlan`; don't add one until an experiment
justifies it. The last grid of a set may be partially filled — that is visible
and honest, unlike silently stretching the sampling interval.

Badge times count from the window start on **one shared clock across all grids
of a set**, so cross-grid boundary reasoning needs no arithmetic.

Token cost is estimated as one visual token per 28×28 px region
(Qwen2/3-VL: 14 px ViT patches, 2×2 merged), summed over grids. It is a
comparison metric, not an exact count for any serving setup. JPEG file size is
deliberately **not** a metric — it does not predict VLM context cost.

## Running a comparison

```bash
uv run python scripts/compare_grid_configs.py --index 0                  # segment ± pad
uv run python scripts/compare_grid_configs.py --index 0 --scope episode  # whole episode
uv run python scripts/compare_grid_configs.py --index 0 --experiment experiments/tile_floor.json
```

Custom configurations come from a source-controlled experiment file
(`experiments/*.json`, validated `ExperimentConfig` — see
docs/configuration.md), never from editing the defaults in source.

Output: `data/samples/grid_experiments/<example_id>/<scope>/<experiment>/`
(experiment `candidates` when none is given) with one directory per config
(`grid_*.jpg`, `metrics.json`), a `comparison.json` recording the experiment +
resolved configs, and an `index.html` that shows all configs at natural pixel
size for side-by-side inspection.

## Candidate configs (starting points, not conclusions)

Source is 640×480 @ 30 fps AV1 (verified by ffprobe, recorded per-run in
metrics). The defaults hold per-grid pixel area (≈ token cost) roughly constant
at ~1.0–1.2 MP while frame density varies, isolating the coverage/legibility
trade-off:

| name | grid | fps | tile (px) | downscale | full grid (px) | ~tok/grid |
|---|---|---|---|---|---|---|
| a_12f_tile336_1fps | 3×4 | 1 | 336×252 | 0.525 | 1344×756 | ~1330 |
| b_25f_tile256_2fps | 5×5 | 2 | 256×192 | 0.400 | 1280×960 | ~1610 |
| c_36f_tile208_3fps | 6×6 | 3 | 208×156 | 0.325 | 1248×936 | ~1518 |
| d_64f_tile160_4fps | 8×8 | 4 | 160×120 | 0.250 | 1280×960 | ~1610 |

## First observations (2026-08-11, one episode — preliminary, n=1)

Example `410_g1_ir/clear_plates/episode_000617/step_005` (16.4 s episode,
2 s subtask "right hand pick up blue plate"), inspected by eye at natural size:

- **a (336 px, 1 fps)**: best per-frame detail, but 1 fps gives a 2 s subtask
  only ~3 frames — temporal resolution, not spatial, is the binding constraint
  for short subtasks.
- **b (256 px, 2 fps)**: plate identity, gripper contact state, and
  rack-seating all clearly judgeable; ~2× the temporal samples of (a) for
  ~+12 % estimated tokens in the segment-scope run. Looks like the current
  sweet-spot region.
- **c (208 px, 3 fps)**: still mostly judgeable; source motion blur (the robot
  moves fast) destroys more detail than the downscale does at this size.
- **d (160 px, 4 fps)**: an entire 16 s episode fits in one ~1840-token grid
  and object colours/gross motion survive, but grasp-vs-approach and
  peg-seating judgements become unreliable. Likely below the floor for
  *evaluation* use; possibly useful as a cheap global overview alongside a
  denser local view.
- Whole-episode coverage cost ~1.8–2.3 k estimated tokens per config here —
  small enough that episode-scope context is plausible for short episodes.

These are observations from ONE episode; nothing is final. Re-run across more
tasks (small objects — eggs, scissors — will punish small tiles harder than
plates do) before changing pipeline defaults. `prepare_example.py` keeps the
proven 12-frame / 336 px geometry until the experiments justify a change.
