from __future__ import annotations

import copy

import pytest

from aimx.research.context import compile_context
from aimx.research.errors import BudgetError, ValidationError
from aimx.research.models import ResearchState, json_bytes


def _encoded_response_size(payload: dict) -> int:
    import json
    return len((json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


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
    # Full response metadata, evidence refs, notes and LF count toward budget.
    budget = 4096

    first = compile_context(state, "improve low-data accuracy", budget)
    second = compile_context(state, "improve low-data accuracy", budget)

    assert first == second
    assert [(item["kind"], item["id"]) for item in first["items"]] == [
        ("finding", "f_relevant")
    ]
    assert first["budget_used"] == _encoded_response_size(first) <= budget


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
    required_items = sorted(items, key=lambda item: (item["kind"], item["id"]))
    required = 4096
    payload = compile_context(state, "accuracy", required)
    full_required = _encoded_response_size(payload)

    with pytest.raises(BudgetError) as error:
        compile_context(state, "accuracy", full_required - 1)
    assert error.value.code == "budget_too_small_for_required_context"
    assert error.value.details["required_bytes"] == full_required
    assert error.value.details["required_group_ids"] == ["f_claim"]

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
    assert len(payload["evidence_refs"]) == 1
    assert payload["evidence_refs"][0]["run_hash"] == ref["run_hash"]
    assert {use["finding_id"] for use in payload["evidence_refs"][0]["uses"]} == {"f_one", "f_two"}


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
        annotations={
            "ann_frontier": {
                "id": "ann_frontier",
                "target_kind": "frontier_item",
                "target_id": "front_1",
                "text": "Pause until the baseline is verified.",
                "author": {"kind": "human", "name": "reviewer"},
            },
            "ann_agenda": {
                "id": "ann_agenda",
                "target_kind": "agenda_item",
                "target_id": "agenda_1",
                "text": "Use the corrected seed list.",
                "author": {"kind": "human", "name": "reviewer"},
            },
        },
    )
    payload = compile_context(state, "unrelated objective", 10_000)
    assert {item["id"] for item in payload["items"]} == {
        "lane_1",
        "front_1",
        "agenda_1",
        "ann_frontier",
        "ann_agenda",
    }


def test_recursive_closure_includes_cross_loop_dependencies_and_marks_them() -> None:
    state = ResearchState(
        activities={
            "act_loop_a": {"id": "act_loop_a", "kind": "loop", "loop_id": "loop-a"},
            "act_loop_b": {"id": "act_loop_b", "kind": "loop", "loop_id": "loop-b"},
        },
        findings={
            "f_seed": {**_finding("f_seed", "accuracy improvement", revision=1), "source_activity_id": "act_loop_a"},
            "f_fix": {**_finding("f_fix", "seed sensitivity correction", revision=4), "source_activity_id": "act_loop_b"},
            "f_final": {**_finding("f_final", "robustness boundary", revision=7), "source_activity_id": "act_loop_a"},
            "f_unrelated": {**_finding("f_unrelated", "accuracy accepted deployment", governance="accepted"), "source_activity_id": "act_loop_b"},
        },
        relations={
            "rel_one": {"id": "rel_one", "source_finding_id": "f_seed", "target_finding_id": "f_fix", "type": "challenges", "active": True, "created_revision": 5},
            "rel_two": {"id": "rel_two", "source_finding_id": "f_fix", "target_finding_id": "f_final", "type": "refines", "active": True, "created_revision": 8},
            "rel_cycle": {"id": "rel_cycle", "source_finding_id": "f_final", "target_finding_id": "f_seed", "type": "supersedes", "active": True, "created_revision": 9},
        },
    )
    payload = compile_context(state, "accuracy improvement", 20_000, loop_id="loop-a")
    finding_ids = {item["id"] for item in payload["items"] if item["kind"] == "finding"}
    assert finding_ids == {"f_seed", "f_fix", "f_final"}
    selected = payload["selection_notes"]["selected_groups"][0]
    assert selected["closure_dependency_ids"] == ["f_fix"]
    assert "f_unrelated" not in finding_ids


def test_correction_closure_keeps_full_chain_cycles_and_relation_annotations() -> None:
    state = ResearchState(
        revision=6,
        findings={
            "f_a": _finding("f_a", "accuracy improves", revision=1),
            "f_b": _finding("f_b", "audit shows unstable accuracy", revision=3),
            "f_c": _finding("f_c", "final accuracy claim is limited", revision=5),
        },
        relations={
            "r_ab": {"id": "r_ab", "source_finding_id": "f_a", "target_finding_id": "f_b", "type": "challenges", "active": True, "created_revision": 4},
            "r_bc": {"id": "r_bc", "source_finding_id": "f_b", "target_finding_id": "f_c", "type": "refines", "active": True, "created_revision": 6},
            "r_ca": {"id": "r_ca", "source_finding_id": "f_c", "target_finding_id": "f_a", "type": "supersedes", "active": True, "created_revision": 6},
        },
        annotations={"ann_relation": {"id": "ann_relation", "target_kind": "relation", "target_id": "r_bc", "text": "the final sample window is smaller", "created_revision": 6}},
    )
    payload = compile_context(state, "accuracy improves", 20000)
    assert {item["id"] for item in payload["items"] if item["kind"] == "finding"} == {"f_a", "f_b", "f_c"}
    assert {item["id"] for item in payload["items"] if item["kind"] == "relation"} == {"r_ab", "r_bc", "r_ca"}
    assert "ann_relation" in {item["id"] for item in payload["items"] if item["kind"] == "annotation"}


def test_legacy_annotation_only_matching_and_unmatched_accepted_findings() -> None:
    state = ResearchState(
        findings={"f_accepted": _finding("f_accepted", "deploy model", governance="accepted")},
        annotations={
            "ann_old": {"id": "ann_old", "target_kind": "finding", "target_id": "f_missing", "text": "accuracy rises on early seeds", "created_revision": 2},
            "ann_fix": {"id": "ann_fix", "target_kind": "finding", "target_id": "f_missing", "text": "accuracy gain disappears under longer window", "created_revision": 9},
        },
    )
    payload = compile_context(state, "accuracy longer window", 10_000)
    assert {item["id"] for item in payload["items"] if item["kind"] == "annotation"} == {"ann_old", "ann_fix"}
    assert not any(item["id"] == "f_accepted" for item in payload["items"])
    assert any("legacy_relationships_unverified" in text for text in payload["selection_notes"]["limitations"])
    assert payload["selection_notes"]["selected_groups"][0]["seed_ids"] == ["ann_fix"]


def test_loop_filter_with_annotation_seed_includes_related_follow_up_finding() -> None:
    state = ResearchState(
        activities={
            "loop": {"id": "loop", "kind": "loop", "loop_id": "loop-1"},
            "follow": {"id": "follow", "kind": "follow_up", "related_loop_id": "loop-1"},
        },
        findings={"later": {**_finding("later", "long horizon accuracy limit", revision=4), "source_activity_id": "follow"}},
        annotations={"legacy": {"id": "legacy", "target_kind": "finding", "target_id": "later", "text": "accuracy needs longer horizon follow-up", "source_activity_id": "follow", "created_revision": 5}},
    )
    payload = compile_context(state, "accuracy horizon", 10000, loop_id="loop-1")
    assert {item["id"] for item in payload["items"] if item["kind"] == "finding"} == {"later"}
    assert "legacy" in {item["id"] for item in payload["items"] if item["kind"] == "annotation"}


def test_loop_annotation_seed_does_not_import_unrelated_activity_annotations() -> None:
    state = ResearchState(
        activities={
            "loop": {"id": "loop", "kind": "loop", "loop_id": "loop-1"},
            "other": {"id": "other", "kind": "analysis"},
        },
        findings={"f_loop": {**_finding("f_loop", "candidate accuracy", revision=1), "source_activity_id": "loop"}},
        annotations={
            "ann_loop": {"id": "ann_loop", "target_kind": "finding", "target_id": "f_loop", "source_activity_id": "loop", "text": "accuracy was measured in the candidate window", "created_revision": 2},
            "ann_other": {"id": "ann_other", "target_kind": "finding", "target_id": "f_loop", "source_activity_id": "other", "text": "accuracy unrelated private note", "created_revision": 3},
        },
    )
    payload = compile_context(state, "accuracy candidate", 10000, loop_id="loop-1")
    annotation_ids = {item["id"] for item in payload["items"] if item["kind"] == "annotation"}
    assert annotation_ids == {"ann_loop"}


def test_omitted_relevance_groups_are_counted_with_stable_reason() -> None:
    state = ResearchState(findings={
        "f_primary": _finding("f_primary", "accuracy improves greatly across many validation cases"),
        "f_second": _finding("f_second", "accuracy declines on a small subset", revision=2),
    })
    large = compile_context(state, "accuracy", 10_000)
    primary_only_size = _encoded_response_size({
        **large,
        "items": [item for item in large["items"] if item["id"] == "f_primary"],
        "evidence_refs": [],
        "selection_notes": {**large["selection_notes"], "selected_groups": [large["selection_notes"]["selected_groups"][0]], "omitted_count": 1, "omitted_by_reason": {"response_budget": 1}},
    })
    assert primary_only_size < 10_000
    assert large["selection_notes"]["omitted_count"] == 0


def test_ten_round_history_keeps_latest_scope_correction_within_budget() -> None:
    annotations = {}
    for round_number in range(1, 11):
        annotation_id = f"ann_{round_number:02d}"
        text = (
            "Candidate improves the local observed accuracy window."
            if round_number < 9
            else "Audit finds seed sensitivity and no stable added gain in the extended historical window; future generalization remains unproven."
        )
        annotations[annotation_id] = {
            "id": annotation_id,
            "target_kind": "research_state",
            "target_id": None,
            "text": text,
            "created_revision": round_number,
        }
    state = ResearchState(revision=10, annotations=annotations)
    payload = compile_context(state, "accuracy extended historical window generalization", 18_000)
    selected_ids = {item["id"] for item in payload["items"] if item["kind"] == "annotation"}
    assert {"ann_09", "ann_10"}.issubset(selected_ids)
    assert payload["budget_used"] == _encoded_response_size(payload)
    repeat = compile_context(state, "accuracy extended historical window generalization", 18_000)
    assert payload == repeat


def test_loop_context_excludes_unrelated_annotations_on_related_governance_items() -> None:
    from aimx.research.context import compile_context
    from aimx.research.models import ResearchState

    state = ResearchState(
        activities={"loop": {"id": "loop", "kind": "loop", "loop_id": "loop-1"}, "other": {"id": "other", "kind": "analysis"}},
        findings={"finding": {**_finding("finding", "candidate accuracy"), "source_activity_id": "loop"}},
        lanes={"lane": {"id": "lane", "name": "research"}},
        frontier_items={"frontier": {"id": "frontier", "lane_id": "lane", "supporting_finding_ids": ["finding"], "status": "active"}},
        annotations={"allowed": {"id": "allowed", "target_kind": "frontier_item", "target_id": "frontier", "source_activity_id": "loop", "text": "candidate direction"}, "unrelated": {"id": "unrelated", "target_kind": "frontier_item", "target_id": "frontier", "source_activity_id": "other", "text": "private note"}},
    )
    payload = compile_context(state, "candidate accuracy", 10000, loop_id="loop-1")
    annotation_ids = {item["id"] for item in payload["items"] if item["kind"] == "annotation"}
    assert annotation_ids == {"allowed"}


@pytest.mark.parametrize(("location", "expected"), [("report.json", "available"), ("missing.json", "missing"), ("../outside.json", "unsupported")])
def test_context_local_artifact_path_diagnostics_are_metadata_only(tmp_path, location: str, expected: str) -> None:
    from aimx.research.context import compile_context
    from aimx.research.models import ResearchState

    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "report.json"
    target.write_text("outside", encoding="utf-8")
    (repo / "report.json").write_text("private", encoding="utf-8")
    (repo / "linked").symlink_to(outside, target_is_directory=True)
    finding = {"id": "finding-1", "claim": "candidate accuracy", "evidence": [{"kind": "local_artifact", "location": location, "sha256": "a" * 64, "size_bytes": 7}]}
    payload = compile_context(ResearchState(findings={"finding-1": finding}), "candidate accuracy", 10000, repo=str(repo))
    assert payload["evidence_refs"][0]["availability"] == expected
    assert payload["evidence_refs"][0]["identity_status"] == "not_checked"


def test_context_local_artifact_symlink_availability_is_unknown_without_following_it(tmp_path) -> None:
    from aimx.research.context import compile_context
    from aimx.research.models import ResearchState

    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "report.json").write_text("private", encoding="utf-8")
    (repo / "linked").symlink_to(outside, target_is_directory=True)
    finding = {"id": "finding-1", "claim": "candidate accuracy", "evidence": [{"kind": "local_artifact", "location": "linked/report.json", "sha256": "a" * 64, "size_bytes": 7}]}
    payload = compile_context(ResearchState(findings={"finding-1": finding}), "candidate accuracy", 10000, repo=str(repo))
    assert payload["evidence_refs"][0]["availability"] == "unknown"
    assert payload["evidence_refs"][0]["identity_status"] == "not_checked"


@pytest.mark.parametrize("objective,budget", [("", 100), ("x", 0), ("x", -1)])
def test_invalid_context_arguments_use_validation_errors(objective: str, budget: int) -> None:
    with pytest.raises(ValidationError):
        compile_context(ResearchState(), objective, budget)
