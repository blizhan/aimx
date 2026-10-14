from __future__ import annotations

import pytest

from aimx.research.errors import ValidationError
from aimx.research.models import ResearchState, ResearchUpdate
from aimx.research.validation import compile_update


def _compile(payload: dict):
    return compile_update(
        ResearchUpdate.from_dict(payload),
        ResearchState(),
        revision=1,
        commit_id="commit_test",
    )


def test_local_references_resolve_in_order_and_created_ids_are_returned() -> None:
    compiled = _compile(
        {
            "schema_version": 1,
            "base_revision": 0,
            "author": {"kind": "agent", "name": "agent-a"},
            "operations": [
                {"op": "finding.create", "local_id": "f1", "claim": "A claim."},
                {"op": "finding.create", "local_id": "f2", "claim": "Another claim."},
                {
                    "op": "relation.create",
                    "local_id": "r1",
                    "source": "$f1",
                    "target": "$f2",
                    "relation_type": "supports",
                },
            ],
        }
    )
    assert compiled.created_ids["f1"].startswith("f_")


def test_self_relation_is_rejected_before_commit() -> None:
    with pytest.raises(ValidationError, match="cannot connect"):
        _compile(
            {
                "schema_version": 1,
                "base_revision": 0,
                "author": {"kind": "agent", "name": "agent-a"},
                "operations": [
                    {"op": "finding.create", "local_id": "f1", "claim": "A claim."},
                    {
                        "op": "relation.create",
                        "source": "$f1",
                        "target": "$f1",
                        "relation_type": "supports",
                    },
                ],
            }
        )


def test_duplicate_local_ids_and_unknown_references_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Duplicate local_id"):
        _compile(
            {
                "schema_version": 1,
                "base_revision": 0,
                "author": {"kind": "agent", "name": "agent-a"},
                "operations": [
                    {"op": "finding.create", "local_id": "f1", "claim": "one"},
                    {"op": "finding.create", "local_id": "f1", "claim": "two"},
                ],
            }
        )

    with pytest.raises(ValidationError, match="Unknown same-update reference"):
        _compile(
            {
                "schema_version": 1,
                "base_revision": 0,
                "author": {"kind": "agent", "name": "agent-a"},
                "operations": [
                    {"op": "relation.create", "source": "$later", "target": "$other", "relation_type": "supports"},
                ],
            }
        )


def test_claim_meaning_cannot_be_overwritten() -> None:
    state = ResearchState(findings={"f_existing": {"id": "f_existing", "claim": "old"}})
    update = ResearchUpdate.from_dict(
        {
            "schema_version": 1,
            "base_revision": 0,
            "author": {"kind": "human", "name": "reviewer"},
            "operations": [
                {
                    "op": "finding.update",
                    "finding_id": "f_existing",
                    "claim": "new",
                }
            ],
        }
    )
    with pytest.raises(ValidationError, match="Unsupported research operation"):
        compile_update(update, state, revision=1, commit_id="commit_test")


def test_loop_activity_checkpoint_must_match_standard_state_repository(tmp_path) -> None:
    import json

    checkpoint = tmp_path / ".aimx/autoresearch/loop-1/state.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"schema_version": 1, "loop_id": "loop-1", "status": "running"}), encoding="utf-8")
    valid = {
        "schema_version": 1, "base_revision": 0, "author": {"kind": "agent", "name": "lead"},
        "operations": [{"op": "activity.create", "kind": "loop", "loop_id": "loop-1", "title": "Loop", "objective": "Objective", "source": {"kind": "fixture", "location": "state.json"}, "checkpoint": {"path": str(checkpoint), "repo": str(tmp_path), "loop_id": "loop-1"}}],
    }
    compiled = compile_update(ResearchUpdate.from_dict(valid), ResearchState(), revision=1, commit_id="commit", repo_root=tmp_path)
    assert compiled.events[0]["payload"]["loop_id"] == "loop-1"
    without_repo_context = compile_update(ResearchUpdate.from_dict(valid), ResearchState(), revision=1, commit_id="commit")
    assert without_repo_context.events[0]["payload"]["loop_id"] == "loop-1"

    alternate = tmp_path / "external" / "state.json"
    alternate.parent.mkdir()
    alternate.write_text(checkpoint.read_text(encoding="utf-8"), encoding="utf-8")
    malformed = {**valid, "operations": [{**valid["operations"][0], "checkpoint": {"path": str(alternate), "repo": str(tmp_path), "loop_id": "loop-1"}}]}
    with pytest.raises(ValidationError, match="standard repository checkpoint location"):
        compile_update(ResearchUpdate.from_dict(malformed), ResearchState(), revision=1, commit_id="commit", repo_root=tmp_path)

    foreign_repo = tmp_path / "other"
    foreign_repo.mkdir()
    malformed = {**valid, "operations": [{**valid["operations"][0], "checkpoint": {"path": str(checkpoint), "repo": str(foreign_repo), "loop_id": "loop-1"}}]}
    with pytest.raises(ValidationError, match="does not match the Research State repository"):
        compile_update(ResearchUpdate.from_dict(malformed), ResearchState(), revision=1, commit_id="commit", repo_root=tmp_path)


def test_explicit_annotation_provenance_is_preserved_and_unresolved_source_rejected() -> None:
    from aimx.research.validation import compile_update

    state = ResearchState(activities={"act": {"id": "act", "kind": "analysis"}}, annotations={"ann": {"id": "ann", "text": "historical interpretation"}})
    valid = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 0, "author": {"kind": "human", "name": "lead"},
        "operations": [{"op": "finding.create", "claim": "Explicitly extracted", "source_activity_id": "act", "topics": ["history"], "source_annotation_ids": ["ann"]}],
    })
    compiled = compile_update(valid, state, revision=1, commit_id="commit_test")
    assert compiled.events[0]["payload"]["source_annotation_ids"] == ["ann"]
    invalid = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 0, "author": {"kind": "human", "name": "lead"},
        "operations": [{"op": "finding.create", "claim": "Bad extraction", "source_annotation_ids": ["ann_missing"]}],
    })
    with pytest.raises(ValidationError, match="Unknown annotation"):
        compile_update(invalid, state, revision=1, commit_id="commit_test")


def test_source_activity_extension_is_optional_and_strict_when_provided() -> None:
    legacy = _compile({
        "schema_version": 1, "base_revision": 0, "author": {"kind": "agent", "name": "agent-a"},
        "operations": [{"op": "finding.create", "local_id": "f1", "claim": "legacy"}],
    })
    assert legacy.events[0]["payload"].get("source_activity_id") is None

    with pytest.raises(ValidationError, match="Unknown activity"):
        _compile({
            "schema_version": 1, "base_revision": 0, "author": {"kind": "agent", "name": "agent-a"},
            "operations": [{"op": "finding.create", "claim": "new", "source_activity_id": "act_missing"}],
        })


def test_structured_finding_requires_complete_semantics_but_legacy_remains_valid() -> None:
    with pytest.raises(ValidationError) as error:
        _compile({
            "schema_version": 1, "base_revision": 0, "author": {"kind": "agent", "name": "agent-a"},
            "operations": [{"op": "finding.create", "claim": "incomplete", "result_kind": "improvement"}],
        })
    assert error.value.code == "incomplete_finding"


def test_agenda_transition_rules_are_enforced() -> None:
    state = ResearchState(
        findings={"f1": {"id": "f1"}},
        agenda_items={"agenda_1": {"id": "agenda_1", "status": "completed"}},
    )
    update = ResearchUpdate.from_dict(
        {
            "schema_version": 1,
            "base_revision": 0,
            "author": {"kind": "agent", "name": "agent-a"},
            "operations": [
                {
                    "op": "agenda.item.transition",
                    "agenda_item_id": "agenda_1",
                    "agenda_status": "active",
                }
            ],
        }
    )
    with pytest.raises(ValidationError) as error:
        compile_update(update, state, revision=1, commit_id="commit_test")
    assert error.value.code == "invalid_transition"


def test_independent_confirmation_rejects_unknown_observation_and_allows_exploratory() -> None:
    from aimx.research.validation import _validate_experiment_contract

    activity = {"id": "act_analysis", "kind": "analysis"}
    scope = {"id": "ds_holdout", "coverage": "unknown", "initial_observation": "unknown", "overlap_keys": ["holdout"], "source": {"kind": "test", "location": "fixture"}}
    state = ResearchState(activities={"act_analysis": activity}, dataset_scopes={"ds_holdout": scope})

    class Resolver:
        def resolve_artifact(self, location: str):
            from pathlib import Path
            return Path(location), "c" * 64, 1

    resolver = Resolver()
    local_refs: dict[str, tuple[str, str]] = {}
    contract = {
        "contract_version": 1,
        "question": "Does the candidate generalize?",
        "interventions": ["candidate"],
        "fixed_controls": ["same evaluator"],
        "expected_evidence": ["paired score"],
        "disconfirmation": ["threshold not met"],
        "confounders": [],
        "evaluation": {
            "protocol_id": "eval-v1",
            "protocol_sha256": "a" * 64,
            "argv": ["uv", "run", "python", "evaluate.py"],
            "metric": "score",
            "acceptance_rule": "score threshold",
            "preflight_evidence": [{"kind": "local_artifact", "location": "preflight.json", "sha256": "c" * 64, "size_bytes": 1, "evidence_type": "preflight", "purpose": "ready report", "source_activity_id": "act_analysis"}],
        },
        "data": {"scope_ids": ["ds_holdout"], "phase": "independent_confirmation", "candidate_fingerprint": "candidate-v1", "observation_revision": 0},
        "resources": [{"kind": "calls", "unit": "count", "planned": 1, "verification": "unverified", "enforcement": "declared_only"}],
        "failure_policy": "stop_pending",
    }
    with pytest.raises(ValidationError) as rejected:
        _validate_experiment_contract(contract, state, local_refs, resolver)
    assert rejected.value.code == "independence_claim_rejected"
    assert rejected.value.details["source"] == scope["source"]
    assert rejected.value.details["query_commands"] == ["aimx research state --json"]

    exploratory = {**contract, "data": {**contract["data"], "phase": "exploratory"}}
    accepted = _validate_experiment_contract(exploratory, state, local_refs, resolver)
    assert accepted["data"]["phase"] == "exploratory"


def test_experiment_contract_rejects_unverified_extensions_and_joint_intervention() -> None:
    from aimx.research.validation import _validate_experiment_contract

    state = ResearchState(activities={"activity": {"id": "activity", "kind": "analysis"}})
    base = {
        "contract_version": 1, "question": "question", "interventions": ["a", "b"],
        "fixed_controls": ["same"], "expected_evidence": ["score"], "disconfirmation": ["threshold"],
        "confounders": [], "evaluation": {"protocol_id": "p", "protocol_sha256": "a" * 64, "argv": ["python", "eval.py"], "metric": "m", "acceptance_rule": "r", "preflight_evidence": [{"kind": "local_artifact", "location": "preflight.json", "sha256": "c" * 64, "size_bytes": 1, "evidence_type": "preflight", "purpose": "ready report", "source_activity_id": "activity"}]},
        "data": {"scope_ids": [], "phase": "exploratory", "candidate_fingerprint": "candidate", "observation_revision": 0},
        "resources": [{"kind": "calls", "unit": "count", "planned": 1, "verification": "unverified", "enforcement": "declared_only"}],
        "failure_policy": "stop_pending",
    }
    with pytest.raises(ValidationError, match="confounder explanation"):
        _validate_experiment_contract(base, state, {}, None)
    complete = {**base, "confounders": ["joint treatment effect"]}
    with pytest.raises(ValidationError, match="extensions must be an object"):
        _validate_experiment_contract({**complete, "extensions": []}, state, {}, None)
    class Resolver:
        def resolve_artifact(self, location: str):
            from pathlib import Path
            return Path(location), "c" * 64, 1

    with pytest.raises(ValidationError, match="attested against a cited preflight artifact"):
        _validate_experiment_contract({**complete, "failure_policy": "continue_independent", "extensions": {"host_failure_policy_verified": True}}, state, {}, Resolver())
    valid = _validate_experiment_contract({**complete, "failure_policy": "continue_independent", "extensions": {"project": "fixture", "host_failure_policy_verified": True, "host_failure_policy_preflight_sha256": "c" * 64}}, state, {}, Resolver())
    assert valid["extensions"] == {"project": "fixture", "host_failure_policy_verified": True, "host_failure_policy_preflight_sha256": "c" * 64}


def test_independent_confirmation_requires_local_preflight_identity_and_scope_coverage() -> None:
    from aimx.research.validation import _validate_experiment_contract

    state = ResearchState(
        activities={"activity": {"id": "activity", "kind": "analysis"}},
        dataset_scopes={"ds": {"id": "ds", "coverage": "unknown", "initial_observation": "unknown", "overlap_keys": ["holdout"], "source": {"kind": "test", "location": "scope.json"}}},
        observations={"freeze": {"id": "freeze", "dataset_scope_id": "ds", "event_kind": "candidate_frozen", "candidate_fingerprint": "candidate-v1", "occurred_at": "2026-10-13T12:00:00Z", "created_revision": 1}},
        revision=1,
    )
    contract = {
        "contract_version": 1, "question": "Confirm candidate", "interventions": ["candidate"],
        "fixed_controls": ["same evaluator"], "expected_evidence": ["score"], "disconfirmation": ["below threshold"], "confounders": [],
        "evaluation": {"protocol_id": "eval-v1", "protocol_sha256": "a" * 64, "argv": ["python", "eval.py"], "metric": "score", "acceptance_rule": "threshold", "preflight_evidence": [{"kind": "aim_run", "run_hash": "b" * 32}]},
        "data": {"scope_ids": ["ds"], "phase": "independent_confirmation", "candidate_fingerprint": "candidate-v1", "observation_revision": 1},
        "resources": [{"kind": "calls", "unit": "count", "planned": 1, "verification": "unverified", "enforcement": "declared_only"}],
        "failure_policy": "stop_pending",
    }
    with pytest.raises(ValidationError, match="preflight_evidence must include a file-backed"):
        _validate_experiment_contract(contract, state, {}, None)

    valid_contract = {**contract, "evaluation": {**contract["evaluation"], "preflight_evidence": [{"kind": "local_artifact", "location": "preflight.json", "sha256": "c" * 64, "size_bytes": 1, "evidence_type": "preflight", "purpose": "ready check", "source_activity_id": "activity"}]}}
    stale_snapshot = {**valid_contract, "data": {**valid_contract["data"], "observation_revision": 0}}

    class Resolver:
        def resolve_artifact(self, location: str):
            from pathlib import Path
            return Path(location), "c" * 64, 1

    with pytest.raises(ValidationError) as stale:
        _validate_experiment_contract(stale_snapshot, state, {}, Resolver())
    assert stale.value.code == "independence_claim_rejected"
    assert stale.value.details["observed_revision"] == state.revision

    with pytest.raises(ValidationError, match="scope is observed or coverage is unknown"):
        _validate_experiment_contract(valid_contract, state, {}, Resolver())


def test_dataset_observations_require_frozen_candidate_and_actual_resource_counts() -> None:
    from aimx.research.validation import _validate_resources

    with pytest.raises(ValidationError, match="actual count"):
        _validate_resources([{"kind": "calls", "unit": "count", "planned": 1, "verification": "verified", "enforcement": "declared_only", "source": "executor log"}])
    assert _validate_resources([{"kind": "calls", "unit": "count", "planned": 1, "actual": 1, "verification": "verified", "enforcement": "declared_only", "source": "executor log"}])[0]["actual"] == 1

    from aimx.research.validation import compile_update

    state = ResearchState(
        activities={"act": {"id": "act", "kind": "analysis"}},
        dataset_scopes={"ds": {"id": "ds", "coverage": "declared_complete", "initial_observation": "declared_unobserved", "overlap_keys": ["holdout"], "source": {"kind": "test", "location": "fixture"}}},
    )
    update = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 0, "author": {"kind": "agent", "name": "test"},
        "operations": [{"op": "dataset.observation.record", "local_id": "promotion", "dataset_scope_id": "ds", "event_kind": "promoted_to_development", "occurred_at": "2026-10-13T12:00:00Z", "source_activity_id": "act", "source": {"kind": "test", "location": "fixture"}, "evidence": [{"kind": "aim_run", "run_hash": "a" * 32}]}],
    })
    with pytest.raises(ValidationError, match="candidate_fingerprint"):
        compile_update(update, state, revision=1, commit_id="commit_test", evidence_resolver=lambda value: value)


def test_experiment_contract_preserves_extensions_after_validation() -> None:
    from aimx.research.validation import _validate_experiment_contract

    class Resolver:
        def resolve_artifact(self, location: str):
            from pathlib import Path
            return Path(location), "c" * 64, 1

    state = ResearchState(activities={"activity": {"id": "activity", "kind": "analysis"}})
    contract = {
        "contract_version": 1, "question": "question", "interventions": ["one"],
        "fixed_controls": ["fixed"], "expected_evidence": ["report"], "disconfirmation": ["threshold"],
        "confounders": [], "evaluation": {"protocol_id": "p", "protocol_sha256": "a" * 64, "argv": ["python", "eval.py"], "metric": "m", "acceptance_rule": "r", "preflight_evidence": [{"kind": "local_artifact", "location": "preflight.json", "sha256": "c" * 64, "size_bytes": 1, "evidence_type": "preflight", "purpose": "ready", "source_activity_id": "activity"}]},
        "data": {"scope_ids": [], "phase": "exploratory", "candidate_fingerprint": "candidate", "observation_revision": 0},
        "resources": [{"kind": "calls", "unit": "count", "planned": 1, "verification": "unverified", "enforcement": "declared_only"}],
        "failure_policy": "stop_pending", "extensions": {"project": "fixture"},
    }
    normalized = _validate_experiment_contract(contract, state, {}, Resolver())
    assert normalized["extensions"] == {"project": "fixture"}


def test_independent_confirmation_rejects_first_scored_and_overlap_without_first_event() -> None:
    from aimx.research.validation import _validate_experiment_contract

    scopes = {
        "candidate": {"id": "candidate", "coverage": "declared_complete", "initial_observation": "declared_unobserved", "overlap_keys": ["sample-a"], "source": {"kind": "test", "location": "a.json"}},
        "other": {"id": "other", "coverage": "declared_complete", "initial_observation": "declared_unobserved", "overlap_keys": ["sample-a"], "source": {"kind": "test", "location": "b.json"}},
    }
    state = ResearchState(
        revision=4, activities={"act": {"id": "act", "kind": "analysis"}}, dataset_scopes=scopes,
        observations={"freeze": {"id": "freeze", "dataset_scope_id": "candidate", "event_kind": "candidate_frozen", "candidate_fingerprint": "candidate-v1", "occurred_at": "2026-10-13T12:00:00Z", "created_revision": 2}},
    )
    class Resolver:
        def resolve_artifact(self, location: str):
            from pathlib import Path
            return Path(location), "c" * 64, 1

    contract = {
        "contract_version": 1, "question": "confirm", "interventions": ["candidate"], "fixed_controls": ["fixed"],
        "expected_evidence": ["score"], "disconfirmation": ["threshold"], "confounders": [],
        "evaluation": {"protocol_id": "p", "protocol_sha256": "a" * 64, "argv": ["python", "eval.py"], "metric": "m", "acceptance_rule": "r", "preflight_evidence": [{"kind": "local_artifact", "location": "preflight.json", "sha256": "c" * 64, "size_bytes": 1, "evidence_type": "preflight", "purpose": "ready", "source_activity_id": "act"}]},
        "data": {"scope_ids": ["candidate"], "phase": "independent_confirmation", "candidate_fingerprint": "candidate-v1", "observation_revision": 4},
        "resources": [{"kind": "calls", "unit": "count", "planned": 1, "verification": "unverified", "enforcement": "declared_only"}], "failure_policy": "stop_pending",
    }
    with pytest.raises(ValidationError, match="cannot be ruled out") as overlap:
        _validate_experiment_contract(contract, state, {}, Resolver())
    assert overlap.value.details["conflicting_scope_id"] == "other"
    assert "query_commands" in overlap.value.details

    scopes["other"]["overlap_keys"] = ["sample-b"]
    state.observations["scored"] = {"id": "scored", "dataset_scope_id": "candidate", "event_kind": "first_scored", "candidate_fingerprint": "candidate-v1", "occurred_at": "2026-10-13T12:01:00Z", "created_revision": 3}
    with pytest.raises(ValidationError, match="already scored or promoted") as scored:
        _validate_experiment_contract(contract, state, {}, Resolver())
    assert scored.value.details["first_observation"]["id"] == "scored"
    assert scored.value.details["source"] == scopes["candidate"]["source"]
    assert scored.value.details["query_commands"] == ["aimx research state --json"]


def test_promoted_scope_cannot_be_declared_independent_after_conversion() -> None:
    from aimx.research.validation import _validate_experiment_contract

    state = ResearchState(
        revision=3,
        activities={"activity": {"id": "activity", "kind": "analysis"}},
        dataset_scopes={"scope": {"id": "scope", "coverage": "declared_complete", "initial_observation": "declared_unobserved", "overlap_keys": ["holdout"], "source": {"kind": "test", "location": "scope.json"}}},
        observations={"freeze": {"id": "freeze", "dataset_scope_id": "scope", "event_kind": "candidate_frozen", "candidate_fingerprint": "candidate", "created_revision": 1, "occurred_at": "2026-10-13T12:00:00Z"}, "promoted": {"id": "promoted", "dataset_scope_id": "scope", "event_kind": "promoted_to_development", "candidate_fingerprint": "candidate", "created_revision": 2, "occurred_at": "2026-10-13T13:00:00Z"}},
    )
    contract = {"contract_version": 1, "question": "confirm", "interventions": ["candidate"], "fixed_controls": ["same"], "expected_evidence": ["score"], "disconfirmation": ["threshold"], "confounders": [], "evaluation": {"protocol_id": "p", "protocol_sha256": "a" * 64, "argv": ["python", "eval.py"], "metric": "m", "acceptance_rule": "r", "preflight_evidence": [{"kind": "local_artifact", "location": "preflight.json", "sha256": "c" * 64, "size_bytes": 1, "evidence_type": "preflight", "purpose": "ready", "source_activity_id": "activity"}]}, "data": {"scope_ids": ["scope"], "phase": "independent_confirmation", "candidate_fingerprint": "candidate", "observation_revision": 3}, "resources": [{"kind": "calls", "unit": "count", "planned": 1, "verification": "unverified", "enforcement": "declared_only"}], "failure_policy": "stop_pending"}

    class Resolver:
        def resolve_artifact(self, location: str):
            from pathlib import Path
            return Path(location), "c" * 64, 1

    with pytest.raises(ValidationError) as rejected:
        _validate_experiment_contract(contract, state, {}, Resolver())
    assert rejected.value.code == "independence_claim_rejected"
    assert rejected.value.details["first_observation"]["id"] == "promoted"


def test_observation_rejects_timezone_missing_but_preserves_late_event_time() -> None:
    from aimx.research.validation import _parse_observation_time
    from aimx.research.validation import compile_update

    with pytest.raises(ValidationError, match="include a timezone"):
        _parse_observation_time("2026-10-13T12:00:00")
    assert _parse_observation_time("2026-10-13T12:00:00-04:00").isoformat().endswith("+00:00")
    state = ResearchState(
        revision=5,
        activities={"act": {"id": "act", "kind": "analysis"}},
        dataset_scopes={"ds": {"id": "ds", "coverage": "declared_complete", "initial_observation": "declared_unobserved", "overlap_keys": ["scope"], "source": {"kind": "test", "location": "scope.json"}}},
        observations={"newer": {"id": "newer", "dataset_scope_id": "ds", "event_kind": "candidate_frozen", "candidate_fingerprint": "candidate-v1", "occurred_at": "2026-10-13T12:00:00Z", "source": {"kind": "test", "location": "newer.json"}}},
    )
    late = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 5, "author": {"kind": "human", "name": "lead"},
        "operations": [{"op": "dataset.observation.record", "local_id": "late", "dataset_scope_id": "ds", "event_kind": "promoted_to_development", "candidate_fingerprint": "candidate-v1", "occurred_at": "2026-10-12T12:00:00Z", "source_activity_id": "act", "source": {"kind": "test", "location": "late.json"}, "evidence": [{"kind": "aim_run", "run_hash": "a" * 32}]}],
    })
    compiled = compile_update(late, state, revision=6, commit_id="commit_late")
    assert compiled.events[0]["payload"]["occurred_at"] == "2026-10-12T12:00:00Z"
    assert compiled.events[0]["payload"]["created_revision"] == 6


def test_duplicate_attempt_keys_within_one_update_are_rejected_before_commit() -> None:
    activity = {"id": "act", "kind": "analysis"}
    agenda = {"id": "agenda", "objective": "score", "status": "active"}
    state = ResearchState(activities={"act": activity}, agenda_items={"agenda": agenda})
    update = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 0, "author": {"kind": "agent", "name": "lead"},
        "operations": [
            {"op": "execution.outcome.record", "local_id": "first", "source_activity_id": "act", "agenda_item_id": "agenda", "attempt_key": "native-1", "execution_status": "failed", "artifact_status": "partial", "evaluation_status": "unknown", "audit_status": "not_run", "summary": "first attempt", "evidence": [], "resources": []},
            {"op": "execution.outcome.record", "local_id": "second", "source_activity_id": "act", "agenda_item_id": "agenda", "attempt_key": "native-1", "execution_status": "succeeded", "artifact_status": "complete", "evaluation_status": "unknown", "audit_status": "not_run", "summary": "duplicate key", "evidence": [], "resources": []},
        ],
    })
    with pytest.raises(ValidationError) as error:
        compile_update(update, state, revision=1, commit_id="commit_test")
    assert error.value.code == "attempt_conflict"


def test_completed_agenda_item_rejects_repeated_completion() -> None:
    state = ResearchState(
        agenda_items={"agenda_1": {"id": "agenda_1", "status": "completed"}},
    )
    update = ResearchUpdate.from_dict(
        {
            "schema_version": 1,
            "base_revision": 0,
            "author": {"kind": "agent", "name": "agent-a"},
            "operations": [
                {
                    "op": "agenda.item.transition",
                    "agenda_item_id": "agenda_1",
                    "agenda_status": "completed",
                }
            ],
        }
    )

    with pytest.raises(ValidationError) as error:
        compile_update(update, state, revision=1, commit_id="commit_test")
    assert error.value.code == "invalid_transition"
