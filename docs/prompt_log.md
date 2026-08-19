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

## 2026-08-11 17:16 — initial commit
- asked: commit; never list Claude as commit co-author from now on
- done: initial commit 8b76456 (34 files, no Claude trailer); preference saved to memory
- done: push to origin failed — SSH agent refused signing for xichaowork@gmail.com key; needs user-side auth

## 2026-08-11 17:19 — push to GitHub
- asked: retry push (user ran ! git push, SSH auth unblocked)
- done: pushed 8b76456 to origin/main (new branch), upstream tracking set

## 2026-08-11 17:14 — confirm config task done
- asked: whether the configuration-system task was already completed
- done: verified on disk (config.py/experiments.py/experiments/, docs, tests) — 51 tests pass, nothing changed

## 2026-08-11 17:24 — where is global config json
- asked: location of the global config JSON
- done: clarified none exists by design — shared paths are Python+env (config.py::PATHS), only experiment overrides are JSON (experiments/*.json); offered optional config.local.json

## 2026-08-11 18:13 — model-service placeholders + EvaluationInput
- asked: role-keyed model-service config (placeholder endpoints) + input-preparation module (EvaluationInput) + CLI + tests + docs; no inference layer
- done: services.py + model_services.json (subtask_generator/reference_judge/evaluator, endpoints null, ports 9020/9030 as server facts); eval_input.py (EvaluationInput = TaskExample + GridSetPlan + ≥1 VisualInput + placeholder instruction, reuses gridset rendering, relative image paths); scripts/prepare_input.py writes evaluation_input.json + grids/<config>/ into examples/<id>/
- done: 22 new tests (73 pass); verified end-to-end on clear_plates ep617 step5 (segment 1-grid + episode 2-grid runs, grid eyeballed against segment); docs/evaluation_input.md; configuration.md/README/CLAUDE.md updated

## 2026-08-12 11:14 — Qwen SFT format research + export modules
- asked: research official Qwen3-VL training data format, examine our stack, write task→episode→subtask + Qwen-format export modules
- done: verified official format from QwenLM/Qwen3-VL qwen-vl-finetune (image list + human/gpt conversations, one <image> tag per image in order, no system role, dict registration, global --max_pixels; identical to Qwen2.5-VL) → docs/qwen_dataset.md
- done: data/episodes.py (EpisodeSubtasks: task→episode→ordered subtasks, invariants verified on 4 tasks then validated in-model); qwen_export.py (QwenSample/QwenTarget, mandatory supervision-source tags, PLACEHOLDER_TARGET until labels exist, self-contained annotations.json+media/+manifest.json); scripts/export_qwen_dataset.py (sample or --tasks traversal, ≤50-example cap, prints registration snippet)
- done: 26 new tests (99 pass); real exports dev3 (3 records) + cp_ep1 (8 records, 1 episode) verified on mount; CLAUDE.md/README updated

## 2026-08-12 11:29 — LLaMA-Factory as fine-tune target
- asked: considering LLaMA-Factory instead of qwen-vl-finetune
- done: verified LF multimodal format from hiyouga/LLaMA-Factory sources — our records are already valid LF ShareGPT (default from/value/human/gpt tags); no format change needed
- done: added llama_factory_snippet (dataset_info.json entry), script now prints both registrations + LF YAML keys (media_dir, image_max_pixels>=1016064 vs 768² default, template qwen3_vl_nothink, LF>=0.9.4); docs/qwen_dataset.md LF section; README/CLAUDE.md note LF preferred; 101 tests pass

## 2026-08-12 12:05 — how to run baseline model locally
- asked: steps to run Qwen3-VL-4B locally for capability testing (remote GPUs later)
- done: checked hardware (RTX 4090 24GB, 62GB RAM, 341GB disk — bf16 fits); recommended vLLM OpenAI-compatible serve on :9040 + fill evaluator endpoint in model_services.json; no code changes

## 2026-08-12 12:10 — use existing LLaMA-Factory env for baseline
- asked: whether vLLM is still needed given an existing LLaMA-Factory env
- done: advised no — llamafactory-cli api serves the same OpenAI-compatible endpoint; flagged LF-specific image_max_pixels downscaling gotcha at inference + >=0.9.4 version check; plan otherwise unchanged

## 2026-08-12 12:21 — find LLaMA-Factory, download model, verify serving
- asked: locate local LLaMA-Factory env + download Qwen3-VL-4B (user afk)
- done: found conda env llama_factory (LF 0.9.6.dev0, torch 2.13+cu130) with BROKEN editable install (source dir deleted); repaired by shallow-cloning hiyouga/LLaMA-Factory to ~/Dev/LLaMA-Factory + pip install -e --no-deps
- done: Qwen/Qwen3-VL-4B-Instruct verified complete in HF cache (8.88GB, was already mostly cached; first attempt failed on all_proxy=socks:// — retried with it unset)
- done: unrequested but verifying — served via llamafactory-cli api :9040 (qwen3vl_4b_api.yaml with image_max_pixels), probed with dev3 record 0: model read badge clock, grounded verdict, 4.7s, 821 tokens = native res (no downscale); filled evaluator endpoint in model_services.json; server left running

## 2026-08-12 13:40 — walkthrough of session scripts
- asked: teach all the scripts/commands used in this session
- done: explained end to end (env discovery, editable-install repair, proxy fix, HF download+verify, LF api serving, poll loop, OpenAI probe script, token check); no code changes

## 2026-08-12 13:54 — why /v1/models says gpt-3.5-turbo
- asked: why the LF API reports model id gpt-3.5-turbo
- done: explained from LF source (api/app.py:93 — hardcoded placeholder unless API_MODEL_NAME set; request.model ignored for routing); suggested API_MODEL_NAME in launch command

## 2026-08-12 13:56 — explain download vs launch vs LF's role
- asked: how the model was downloaded, how it was launched, and what LLaMA-Factory has to do with either
- done: explained separation (HF Hub cache = weights, transformers = loader, LF = template/preprocessing/API harness); no changes

## 2026-08-12 14:15 — reusable serving-setup playbook
- asked: distil the setup knowledge for future self-service
- done: wrote docs/local_serving.md (weights/loader/harness/client model, env verification, download verification, 3 load-bearing serving choices, layered smoke test, endpoint recording)

## 2026-08-12 14:22 — README structure refresh + regular commits
- asked: update README to reflect the current file structure; make regular commits to keep the repo updated
- done: rewrote README Layout section to mirror the actual tree (scripts enumerated, model_services.json, tests, all docs incl. local_serving/prompt_log, generated data/ dirs); fixed stale "placeholder endpoints" wording for services.py
- done: committed the pending work as logical commits (episodes hierarchy, EvaluationInput, model services, Qwen export, docs); 102 tests passing before committing

## 2026-08-19 14:45 — WRC LinkedIn outreach sheet
- asked: draft LinkedIn connection requests for WRC 2026 main-forum speakers (photos of agenda board)
- done: extracted 14 speakers/panelists from board photos into docs/wrc2026_linkedin_outreach.md with <200-char personalized notes
- done: web-searched each person's LinkedIn; URLs recorded with confidence notes (batch 2 in progress)
