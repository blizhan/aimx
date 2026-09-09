from __future__ import annotations

import copy

import pytest

from aimx.research.context import compile_context
from aimx.research.errors import BudgetError, ValidationError
from aimx.research.models import ResearchState, json_bytes


def _finding(
    finding_id: str,
    claim: str,
    *,
    revision: int = 1,
    evidence: list[dict] | None = None,
    status: str = "candidate",
    governance: str = "proposed",
) -> dict:
    return {
        "id": finding_id,
        "claim": claim,
        "evidence": evidence or [],
        "epistemic_status": status,
        "confidence": "medium",
        "governance_status": governance,
        "created_revision": revision,
        "created_commit_id": f"commit_{revision}",
        "author": {"kind": "agent", "name": "test"},
    }


def test_relevance_and_irrelevant_history_are_deterministic() -> None:
    state = ResearchState(
        revision=3,
        findings={
            "f_relevant": _finding("f_relevant", "accuracy improves on low data"),
            "f_irrelevant": _finding("f_irrelevant", "unrelated deployment detail"),
        },
    )
    relevant = {"kind": "finding", "id": "f_relevant", "data": state.findings["f_relevant"]}
    budget = json_bytes([relevant])

    first = compile_context(state, "improve low-data accuracy", budget)
    second = compile_context(state, "improve low-data accuracy", budget)

    assert first == second
    assert [(item["kind"], item["id"]) for item in first["items"]] == [
        ("finding", "f_relevant")
    ]
    assert first["budget_used"] == json_bytes(first["items"]) <= budget


def test_human_annotation_and_governance_influence_selection() -> None:
    state = ResearchState(
        findings={
            "f_plain": _finding("f_plain", "a claim"),
            "f_steered": _finding("f_steered", "a claim"),
        },
        annotations={
            "ann_1": {
                "id": "ann_1",
                "target_kind": "finding",
                "target_id": "f_steered",
                "text": "focus accuracy next",
                "author": {"kind": "human", "name": "reviewer"},
                "created_revision": 2,
            }
        },
    )
    payload = compile_context(state, "focus accuracy next", 10_000)
    ids = [item["id"] for item in payload["items"] if item["kind"] == "finding"]
    assert ids[0] == "f_steered"
    assert "ann_1" in [item["id"] for item in payload["items"]]


def test_challenge_closure_is_packed_or_fails_as_a_unit() -> None:
    state = ResearchState(
        findings={
            "f_claim": _finding("f_claim", "accuracy improves"),
            "f_counter": _finding("f_counter", "seed counterexample", status="contradicted"),
        },
        relations={
            "rel_challenge": {
                "id": "rel_challenge",
                "source_finding_id": "f_counter",
                "target_finding_id": "f_claim",
                "type": "challenges",
                "active": True,
                "created_revision": 2,
            }
        },
    )
    items = [
        {"kind": "finding", "id": "f_claim", "data": state.findings["f_claim"]},
        {"kind": "finding", "id": "f_counter", "data": state.findings["f_counter"]},
        {"kind": "relation", "id": "rel_challenge", "data": state.relations["rel_challenge"]},
    ]
    required = json_bytes(sorted(items, key=lambda item: (item["kind"], item["id"])))

    with pytest.raises(BudgetError) as error:
        compile_context(state, "accuracy", required - 1)
    assert error.value.code == "budget_too_small_for_required_context"
    assert error.value.details["required_bytes"] == required

    payload = compile_context(state, "accuracy", required)
    assert {item["id"] for item in payload["items"]} == {
        "f_claim",
        "f_counter",
        "rel_challenge",
    }


def test_evidence_references_are_canonicalized_and_deduplicated() -> None:
    ref = {"kind": "aim_run", "run_hash": "a" * 32, "role": "candidate"}
    state = ResearchState(
        findings={
            "f_one": _finding("f_one", "accuracy one", evidence=[ref]),
            "f_two": _finding("f_two", "accuracy two", evidence=[copy.deepcopy(ref)]),
        }
    )
    payload = compile_context(state, "accuracy", 10_000)
    assert payload["evidence_refs"] == [ref]


def test_active_frontier_and_agenda_are_available_without_matching_findings() -> None:
    state = ResearchState(
        lanes={"lane_1": {"id": "lane_1", "name": "promising", "active": True}},
        frontier_items={
            "front_1": {
                "id": "front_1",
                "lane_id": "lane_1",
                "subject": {"kind": "direction", "title": "test seeds"},
                "rationale": "Resolve seed sensitivity.",
                "priority": 80,
                "status": "active",
            }
        },
        agenda_items={
            "agenda_1": {
                "id": "agenda_1",
                "objective": "Run a second seed",
                "status": "proposed",
                "priority": 70,
                "created_revision": 2,
            }
        },
    )
    payload = compile_context(state, "unrelated objective", 10_000)
    assert {item["id"] for item in payload["items"]} == {
        "lane_1",
        "front_1",
        "agenda_1",
    }


@pytest.mark.parametrize("objective,budget", [("", 100), ("x", 0), ("x", -1)])
def test_invalid_context_arguments_use_validation_errors(objective: str, budget: int) -> None:
    with pytest.raises(ValidationError):
        compile_context(ResearchState(), objective, budget)
