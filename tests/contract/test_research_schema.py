from __future__ import annotations

import json
from pathlib import Path

from aimx.research.events import SUPPORTED_OPERATIONS


def test_research_update_schema_matches_supported_operation_catalog() -> None:
    path = Path("specs/007-research-state/contracts/research-update.schema.json")
    schema = json.loads(path.read_text(encoding="utf-8"))
    values = set(schema["$defs"]["operation"]["properties"]["op"]["enum"])
    assert values == set(SUPPORTED_OPERATIONS)
    assert schema["properties"]["schema_version"]["const"] == 1
    assert schema["properties"]["operations"]["minItems"] == 1


def test_schema_keeps_legacy_envelope_and_aim_run_evidence_valid() -> None:
    schema = json.loads(
        Path("specs/007-research-state/contracts/research-update.schema.json").read_text()
    )
    assert schema["required"] == ["schema_version", "base_revision", "author", "operations"]
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["evidence"]["properties"]["kind"]["const"] == "aim_run"
    assert schema["$defs"]["operation"]["properties"]["source_activity_id"] == {
        "$ref": "#/$defs/entityRef"
    }
    assert schema["properties"]["schema_version"]["const"] == 1


def test_schema_exposes_incremental_research_knowledge_operations() -> None:
    schema = json.loads(
        Path("specs/007-research-state/contracts/research-update.schema.json").read_text()
    )
    supported = set(schema["$defs"]["operation"]["properties"]["op"]["enum"])
    assert {
        "activity.create",
        "activity.artifact.add",
        "evidence.review",
        "dataset.scope.create",
        "dataset.observation.record",
        "execution.outcome.record",
    } <= supported
    assert schema["$defs"]["operation"]["additionalProperties"] is False


def test_schema_keeps_legacy_envelope_and_aim_run_evidence_valid() -> None:
    schema = json.loads(
        Path("specs/007-research-state/contracts/research-update.schema.json").read_text()
    )
    assert schema["required"] == ["schema_version", "base_revision", "author", "operations"]
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["evidence"]["oneOf"] == [
        {"$ref": "#/$defs/aimRunEvidence"},
        {"$ref": "#/$defs/localArtifactEvidence"},
    ]
    assert schema["$defs"]["operation"]["properties"]["source_activity_id"]["$ref"] == "#/$defs/entityRef"
    assert schema["properties"]["schema_version"]["const"] == 1


def test_experiment_contract_and_attempt_resources_have_explicit_schema_fields() -> None:
    schema = json.loads(Path("specs/007-research-state/contracts/research-update.schema.json").read_text())
    operation = schema["$defs"]["operation"]
    contract = schema["$defs"]["experimentContract"]
    assert operation["properties"]["experiment_contract"] == {"$ref": "#/$defs/experimentContract"}
    assert set(contract["required"]) == {"contract_version", "question", "interventions", "fixed_controls", "expected_evidence", "disconfirmation", "confounders", "evaluation", "data", "resources", "failure_policy"}
    assert contract["additionalProperties"] is False
    assert contract["properties"]["failure_policy"]["enum"] == ["stop_pending", "continue_independent"]
    extensions = contract["properties"]["extensions"]
    assert extensions["properties"]["host_failure_policy_preflight_sha256"]["pattern"] == "^[0-9a-f]{64}$"
    assert contract["properties"]["evaluation"]["properties"]["preflight_evidence"]["minItems"] == 1
    assert contract["properties"]["data"]["properties"]["phase"]["enum"] == ["exploratory", "independent_confirmation"]
    assert operation["properties"]["occurred_at"]["format"] == "date-time"
    assert operation["properties"]["resources"]["type"] == "array"
    assert operation["properties"]["event_kind"]["enum"] == ["scope_declared", "candidate_frozen", "first_scored", "promoted_to_development"]
    assert operation["properties"]["attempt_id"] == {"type": "string", "minLength": 1}
    assert operation["properties"]["resources"]["items"]["required"] == ["kind", "unit", "planned", "verification", "enforcement"]
    assert operation["properties"]["resources"]["items"]["properties"]["verification"]["enum"] == ["verified", "unverified"]


def test_schema_exposes_incremental_research_knowledge_operations() -> None:
    schema = json.loads(
        Path("specs/007-research-state/contracts/research-update.schema.json").read_text()
    )
    supported = set(schema["$defs"]["operation"]["properties"]["op"]["enum"])
    assert {
        "activity.create",
        "activity.artifact.add",
        "evidence.review",
        "dataset.scope.create",
        "dataset.observation.record",
        "execution.outcome.record",
    } <= supported
    assert schema["$defs"]["operation"]["additionalProperties"] is False
