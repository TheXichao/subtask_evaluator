# Configuration conventions

One principle: **one source of truth where consistency matters; local ownership
where independence matters; explicit overrides where experimentation matters.**
There is no global settings file; each kind of value has exactly one home.

## The four homes

### 1. Shared project/environment paths — `src/subtask_checker/config.py`

`ProjectPaths` (frozen Pydantic model) is the canonical definition of where
things live; `config.PATHS` is the shared default instance, built once from the
environment:

| field | default | env override |
|---|---|---|
| `project_root` | this checkout | — |
| `data_root` | `~/remote/410_g1_ir` (read-only mount) | `SUBTASK_CHECKER_DATA_ROOT` |
| `team_repo_root` | `~/remote/Unitree_Robo_Describer` (read-only mount) | `SUBTASK_CHECKER_TEAM_REPO_ROOT` |
| `evaluation_reference_root` | `~/Dev/evaluation` (reference only) | `SUBTASK_CHECKER_EVALUATION_ROOT` |
| `output_root` | `<repo>/data` (gitignored) | `SUBTASK_CHECKER_OUTPUT_ROOT` |

Only infrastructure identity goes here. Derived shared artifacts used by more
than one script (e.g. `PATHS.dev_sample_jsonl`) are properties; single-script
output locations stay in the script that owns them.

`config.require_writable(path)` refuses any write target under a read-only
root; it is called at the artifact write sites, so processing code cannot
accidentally write into team-owned trees.

### 2. Component configuration — inside the component

A value that matters to one component lives in that component's module as a
Pydantic config model (or a module constant for fixed defaults). Example:
`video/gridset.py::GridConfig` owns grid geometry (`rows`, `columns`,
`sample_fps`, `tile_width`, …) with the component defaults as Pydantic field
defaults. Components never reach into other components' settings.

Functions take their config explicitly (`plan_grid_set(example, config)`), not
via hidden globals — the one sanctioned "default instance" is `config.PATHS`,
and even there every function accepts an explicit override for tests.

### 3. Experiment overrides — `experiments/*.json`

An experiment varies component defaults without touching source code. A file
names only what it varies; Pydantic fills the rest from component defaults and
rejects unknown fields (`extra="forbid"`), so a typo cannot silently configure
nothing:

```json
{
  "name": "tile_floor",
  "description": "why this experiment exists",
  "grids": [
    {"name": "t224_5x5_2fps", "rows": 5, "columns": 5,
     "sample_fps": 2.0, "tile_width": 224}
  ]
}
```

`subtask_checker/experiments.py::ExperimentConfig` validates the file
(`load_experiment(path)`). Experiment files are source-controlled; generated
artifacts record the experiment name and the fully-resolved configs
(`comparison.json`), so every run is reproducible from the repo alone.

```bash
uv run python scripts/compare_grid_configs.py --index 0 --experiment experiments/tile_floor.json
```

Format choice: JSON, because every data contract in this repo is already
JSON/JSONL parsed by Pydantic (`model_validate_json`), and it needs no new
dependency (`tomllib` is 3.11+; we support 3.10). Re-evaluate only if config
files need comments badly.

### 4. Model service roles — `model_services.json`

The inventory of model services around the project, keyed by **role**
(`subtask_generator`, `reference_judge`, `evaluator`), validated by
`services.py::ModelServices` (`extra="forbid"`, so an unknown role is an
error). The JSON file at the repo root is the single source of truth —
`services.load_model_services()` returns the typed object.

`endpoint` is `null` until a service is actually reachable: the known services
run as vLLM on the GPU server behind SSH tunnels, so `server_port` is a
recorded fact about the server, never something to synthesise a URL from.
`ModelService.require_endpoint()` fails loudly while the placeholder stands.
Nothing in the data-preparation layer reads this file; it exists for the
future inference module.

## Adding configuration for a new component (e.g. training, evaluator)

1. Define `SomethingConfig(BaseModel)` **in that component's module** with
   validated fields and sensible defaults. No central registration.
2. Pass it explicitly to the component's functions.
3. If experiments need to vary it, add one optional field on
   `ExperimentConfig` (e.g. `training: TrainingConfig | None = None`).
4. Have artifacts record the resolved config they were produced with.

That is the entire mechanism — no registry, discovery, or inheritance layers.

## What is NOT configuration

- **Domain facts** (the server path prefix, the subtask JSONL location, the
  excluded task dirs) — constants in code (`config.py` top section); no
  experiment may "configure" the dataset's layout.
- **Evaluation rubric / judgement logic** (what makes a segment valid) —
  domain logic in code and prompts, never a config knob.
- **Data schema** (`TaskExample` etc.) — the data contract, not settings.
- **Anything only one function uses once** — a parameter, not configuration.
