import json

import pytest
from pydantic import ValidationError

from subtask_checker.services import (
    MODEL_SERVICES_JSON,
    ModelService,
    ModelServices,
    load_model_services,
)


def test_repo_inventory_loads_with_all_roles():
    services = load_model_services()
    assert services.subtask_generator.model == "Qwen3.6-27B-sft-merged"
    assert services.subtask_generator.server_port == 9020
    assert services.reference_judge.model == "Qwen3-VL-235B-A22B-Instruct"
    assert services.reference_judge.server_port == 9030
    assert services.evaluator.model == "Qwen3-VL-4B-Instruct"


def test_repo_inventory_endpoints_are_placeholder_or_base_url():
    # An endpoint is either an honest null placeholder or a reachable-looking
    # OpenAI-compatible base URL — never a synthesised server-port address
    # (ports on the GPU server are facts, not URLs; see services.py).
    services = load_model_services()
    for svc in (services.subtask_generator, services.reference_judge, services.evaluator):
        assert svc.endpoint is None or svc.endpoint.startswith(("http://", "https://"))
        if svc.server_port is not None and svc.endpoint is not None:
            assert f":{svc.server_port}" not in svc.endpoint or "127.0.0.1" in svc.endpoint


def test_remote_roles_remain_placeholders():
    # The two server-side roles are reachable only through SSH tunnels that do
    # not exist on this machine; their endpoints must stay null until one does.
    services = load_model_services()
    assert services.subtask_generator.endpoint is None
    assert services.reference_judge.endpoint is None


def test_placeholder_endpoint_raises_loudly():
    svc = ModelService(model="qwen3-vl-4b")
    with pytest.raises(ValueError, match="placeholder"):
        svc.require_endpoint()


def test_configured_endpoint_is_returned():
    svc = ModelService(model="m", endpoint="http://localhost:9030/v1")
    assert svc.require_endpoint() == "http://localhost:9030/v1"


def test_unknown_role_rejected(tmp_path):
    data = json.loads(MODEL_SERVICES_JSON.read_text(encoding="utf-8"))
    data["model_7"] = {"model": "mystery"}
    p = tmp_path / "model_services.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValidationError):
        load_model_services(p)


def test_missing_role_rejected(tmp_path):
    data = json.loads(MODEL_SERVICES_JSON.read_text(encoding="utf-8"))
    del data["evaluator"]
    p = tmp_path / "model_services.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValidationError):
        load_model_services(p)


def test_round_trip():
    services = load_model_services()
    assert ModelServices.model_validate_json(services.model_dump_json()) == services
