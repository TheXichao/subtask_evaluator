# Prompt log

One entry per user prompt: what was asked, what happened. Newest at the bottom.

## 2026-08-11 ~13:20 — investigate existing data & pipelines (retro entry)
- asked: investigate 410_g1_ir dataset, team subtasker repo, and ~/Dev/evaluation; report findings, no implementation
- done: confirmed LeRobot v3.0 (33 tasks, 33,653 eps, 15,808 reviewed segments); mapped subtasker + judge + grid pipelines
- done: wrote investigation report + dataset stats sweep (now at docs/investigation_report.md, docs/dataset_stats.json)

## 2026-08-11 16:26 — add prompt-logging rule
- asked: CLAUDE.md rule — after every prompt, log a dated petite summary (asked/done bullets) under docs
- done: added "Prompt log (required)" section to CLAUDE.md; created docs/prompt_log.md with retro + current entries

## 2026-08-11 16:49 — visual-token-efficient grid configs
- asked: make grid density configurable (fps/rows/cols/tile/chunk), optimise info per visual token, render candidate configs side by side with metrics
- done: added video/gridset.py (GridConfig → multi-grid plan + GridSetMetrics incl. 28px-patch token estimate), compute_window/probe_video in frames.py, scaled badges in grid.py
- done: scripts/compare_grid_configs.py renders 4 candidates (12f/336px…64f/160px, ~constant tokens/grid) per example + comparison.json + index.html; 13 new tests (35 pass)
- done: ran on clear_plates ep 617 (segment+episode scope), eyeballed grids; preliminary n=1 findings in docs/grid_experiments.md (256px/2fps looks like sweet spot; 160px marginal)

## 2026-08-11 17:11 — configuration system conventions
- asked: clean config system — canonical shared paths, component-owned config, Pydantic-validated experiment overrides, no global config file
- done: config.py → frozen ProjectPaths/PATHS (repo/mount/team-repo/eval-ref/output roots, SUBTASK_CHECKER_* env overrides) + require_writable guard at write sites
- done: experiments.py (ExperimentConfig, extra=forbid, name/dupe validation) + experiments/tile_floor.json; compare_grid_configs gains --experiment (replaces --config-json), outputs under <scope>/<experiment>/ with provenance
- done: scripts dedup'd onto PATHS; 16 new tests (51 pass); verified tile_floor run end-to-end; docs/configuration.md; README + CLAUDE.md convention section

## 2026-08-11 17:10 — git init + remote
- asked: initialise git with remote git@github.com:TheXichao/subtask_evaluator.git
- done: git init -b main, added origin remote; existing .gitignore verified (data/, .venv, caches excluded); no commit/push made
