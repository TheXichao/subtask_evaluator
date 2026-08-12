"""Model service roles: which model fills each role, and where it is served.

Three roles exist around this project (verified in docs/investigation_report.md):

  * ``subtask_generator`` — Qwen3.6-27B-sft, the team's subtasker that produced
    the pre-review annotations this project judges (vLLM ``:9020`` on the GPU
    server).
  * ``reference_judge`` — Qwen3-VL-235B, the judge the previous evaluation
    project (``~/Dev/evaluation``) ran against (vLLM ``:9030``).
  * ``evaluator`` — Qwen3-VL-4B, the model this project prepares inputs for.
    Not served anywhere yet.

This module only DESCRIBES the services — nothing here opens a connection; a
future inference module will consume :class:`ModelServices`, and the
data-preparation side never needs it. The inventory itself lives in the
source-controlled ``model_services.json`` at the repo root (JSON → Pydantic
validation, same pattern as ``experiments/*.json``); this module validates it.

Endpoints are deliberately ``null`` placeholders. The known services listen on
GPU-server ports reached through SSH tunnels, so a port number is a fact about
the server, NOT a usable URL — never synthesise ``http://10.0.8.204:9020``-style
addresses from it. Fill ``endpoint`` in ``model_services.json`` with a reachable
OpenAI-compatible base URL once a tunnel (or direct route) actually exists.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from subtask_checker import config

MODEL_SERVICES_JSON = config.PATHS.project_root / "model_services.json"


class ModelService(BaseModel):
    """One service role: the model filling it and how (if at all) to reach it."""

    model_config = ConfigDict(extra="forbid")

    model: str
    # Reachable OpenAI-compatible base URL, e.g. once an SSH tunnel is up.
    # None = placeholder: the service is known but not configured on this machine.
    endpoint: str | None = None
    # vLLM port on the GPU server — a recorded fact, not an address to connect to.
    server_port: int | None = None
    notes: str = ""

    def require_endpoint(self) -> str:
        """The endpoint, or a loud failure while it is still a placeholder."""
        if self.endpoint is None:
            raise ValueError(
                f"no endpoint configured for {self.model!r} — it is a placeholder; "
                "fill 'endpoint' in model_services.json once the service is reachable"
            )
        return self.endpoint


class ModelServices(BaseModel):
    """The service roles as explicit fields, so code addresses a service by role
    (``services.evaluator``) and an unknown role is a validation error, never a
    silently-ignored ``model_7``."""

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    subtask_generator: ModelService
    reference_judge: ModelService
    evaluator: ModelService


def load_model_services(path: str | Path = MODEL_SERVICES_JSON) -> ModelServices:
    """Parse and validate the service inventory; raises on anything malformed."""
    return ModelServices.model_validate_json(Path(path).read_text(encoding="utf-8"))
