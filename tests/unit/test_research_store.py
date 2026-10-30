from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from aimx.research.errors import RevisionConflict, StoreError, ValidationError
from aimx.research.models import ResearchUpdate
from aimx.research.store import (
    apply_update,
    current_revision,
    database_path,
    dry_run_update,
    read_commits,
    read_events,
    read_state,
)


def _update(payload: dict) -> ResearchUpdate:
    return ResearchUpdate.from_dict(payload)


def test_missing_store_reads_empty_without_creating_sidecar(research_repo: Path) -> None:
    assert read_state(research_repo).as_dict()["revision"] == 0
    assert not database_path(research_repo).exists()


def test_explicit_write_initializes_store_and_revisions_are_contiguous(
    research_repo: Path, research_update_factory
) -> None:
    first = apply_update(
        research_repo,
        _update(research_update_factory(client_update_id="round-1")),
    )
    second = apply_update(
        research_repo,
        _update(
            research_update_factory(
                base_revision=1,
                operations=[
                    {
                        "op": "annotation.create",
                        "local_id": "note",
                        "annotation_target_kind": "research_state",
                        "text": "Keep the validation split fixed.",
                    }
                ],
                client_update_id="round-2",
            )
        ),
    )

    assert first.revision == 1
    assert second.revision == 2
    assert read_state(research_repo).revision == 2
    assert database_path(research_repo).is_file()
    assert current_revision(research_repo) == 2
    assert len(read_commits(research_repo)) == 2
    assert len(read_events(research_repo)) == 2


def test_old_idempotent_request_fingerprint_is_not_rewritten_by_extensions(
    research_repo: Path, research_update_factory
) -> None:
    import hashlib
    import json

    payload = research_update_factory(client_update_id="legacy-exact")
    fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    apply_update(research_repo, _update(payload))
    with sqlite3.connect(database_path(research_repo)) as connection:
        stored = connection.execute("SELECT request_fingerprint FROM commits WHERE client_update_id = ?", ("legacy-exact",)).fetchone()[0]
    assert stored == fingerprint
    assert apply_update(research_repo, _update(payload)).idempotent


def test_revision_conflict_does_not_add_a_commit(
    research_repo: Path, research_update_factory
) -> None:
    apply_update(research_repo, _update(research_update_factory()))

    with pytest.raises(RevisionConflict) as error:
        apply_update(
            research_repo,
            _update(
                research_update_factory(
                    base_revision=0,
                    client_update_id="stale",
                )
            ),
        )

    assert error.value.exit_status == 3
    assert read_state(research_repo).revision == 1


def test_client_update_id_is_idempotent(research_repo: Path, research_update_factory) -> None:
    update = _update(research_update_factory(client_update_id="same-request"))
    first = apply_update(research_repo, update)
    repeated = apply_update(research_repo, update)

    assert repeated.idempotent is True
    assert repeated.commit_id == first.commit_id
    assert repeated.revision == first.revision == 1
    assert repeated.created_ids == first.created_ids
    assert len(read_state(research_repo).findings) == 1


def test_client_update_id_rejects_a_different_request(
    research_repo: Path, research_update_factory
) -> None:
    first = _update(research_update_factory(client_update_id="same-request"))
    apply_update(research_repo, first)
    different = _update(
        research_update_factory(
            base_revision=1,
            client_update_id="same-request",
            operations=[
                {
                    "op": "finding.create",
                    "local_id": "different",
                    "claim": "A different claim.",
                }
            ],
        )
    )

    with pytest.raises(ValidationError) as error:
        apply_update(research_repo, different)

    assert error.value.code == "idempotency_conflict"
    assert read_state(research_repo).revision == 1


def test_dry_run_validates_without_creating_store(
    research_repo: Path, research_update_factory
) -> None:
    state, created_ids = dry_run_update(
        research_repo,
        _update(research_update_factory()),
    )

    assert state.revision == 0
    assert set(created_ids) == {"finding"}
    assert not database_path(research_repo).exists()


def test_invalid_multi_operation_rolls_back_and_does_not_create_store(
    research_repo: Path, research_update_factory
) -> None:
    update = _update(
        research_update_factory(
            operations=[
                {
                    "op": "finding.create",
                    "local_id": "f1",
                    "claim": "A claim.",
                },
                {
                    "op": "relation.create",
                    "source": "$f1",
                    "target": "$missing",
                    "relation_type": "supports",
                },
            ]
        )
    )

    with pytest.raises(ValidationError, match="Unknown same-update reference"):
        apply_update(research_repo, update)

    assert read_state(research_repo).revision == 0
    assert not database_path(research_repo).exists()


def test_dot_aim_path_resolves_to_repository_root(
    research_repo: Path, research_update_factory
) -> None:
    dot_aim = research_repo / ".aim"
    dot_aim.mkdir()
    result = apply_update(dot_aim, _update(research_update_factory()))

    assert result.revision == 1
    assert database_path(research_repo).is_file()


def test_read_receipt_is_read_only_and_reports_not_found_conflict_and_result(research_repo: Path, research_update_factory) -> None:
    from aimx.research.store import read_receipt, _request_fingerprint

    missing = read_receipt(research_repo, "missing")
    assert missing["status"] == "not_found"
    assert not database_path(research_repo).exists()

    update = _update(research_update_factory(client_update_id="receipt-key"))
    result = apply_update(research_repo, update)
    found = read_receipt(research_repo, "receipt-key", _request_fingerprint(update))
    assert found["status"] == "committed"
    assert found["commit_id"] == result.commit_id
    assert found["revision"] == result.revision
    assert found["request_fingerprint"] == _request_fingerprint(update)
    with pytest.raises(ValidationError) as conflict:
        read_receipt(research_repo, "receipt-key", "0" * 64)
    assert conflict.value.code == "idempotency_conflict"


def test_exact_file_update_replay_survives_artifact_removal(
    research_repo: Path, tmp_path: Path,
) -> None:
    import hashlib
    from aimx.aim_bridge.research_evidence import make_evidence_resolver

    artifact = research_repo / "report.json"
    artifact.write_text('{"result": "durable"}\\n', encoding="utf-8")
    content = artifact.read_bytes()
    update = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 0, "client_update_id": "file-replay",
        "author": {"kind": "human", "name": "lead"},
        "operations": [
            {"op": "activity.create", "local_id": "activity", "kind": "analysis", "title": "Analysis", "objective": "Preserve result", "source": {"kind": "fixture", "location": str(artifact)}},
            {"op": "finding.create", "local_id": "finding", "claim": "Recorded file result", "source_activity_id": "$activity", "topics": ["result"], "result_kind": "insufficient_evidence", "scope": {"description": "fixture", "dataset_scope_ids": [], "conditions": ["same evaluator"]}, "limitations": ["single run"], "evidence": [{"kind": "local_artifact", "location": str(artifact), "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content), "evidence_type": "report", "purpose": "replay test", "source_activity_id": "$activity"}]},
        ],
    })
    first = apply_update(research_repo, update, make_evidence_resolver(research_repo))
    artifact.unlink()
    replayed = apply_update(research_repo, update, make_evidence_resolver(research_repo))
    assert replayed.idempotent is True
    assert replayed.commit_id == first.commit_id
    assert replayed.revision == first.revision == 1
    assert len(read_state(research_repo).findings) == 1


def test_read_receipt_marks_legacy_commit_unverifiable_without_altering_schema(research_repo: Path) -> None:
    from aimx.research.store import read_receipt

    path = database_path(research_repo)
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.executescript("CREATE TABLE commits (id TEXT PRIMARY KEY, revision INTEGER, client_update_id TEXT); INSERT INTO commits VALUES ('c1', 1, 'old-key');")
    before = path.read_bytes()
    assert read_receipt(research_repo, "old-key")["status"] == "receipt_unverifiable"
    assert path.read_bytes() == before


def test_duplicate_attempt_key_in_update_rolls_back_entire_store_write(
    research_repo: Path,
) -> None:
    update = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 0, "author": {"kind": "agent", "name": "lead"},
        "operations": [
            {"op": "activity.create", "local_id": "activity", "kind": "analysis", "title": "Run", "objective": "Score", "source": {"kind": "test", "location": "fixture"}},
            {"op": "agenda.item.create", "local_id": "agenda", "objective": "score", "frontier_item_ids": [], "motivating_finding_ids": ["$finding"]},
            {"op": "finding.create", "local_id": "finding", "claim": "baseline"},
        ],
    })
    # Agenda references must be declared first; create a direction lane and use it as motivation.
    update = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 0, "author": {"kind": "agent", "name": "lead"},
        "operations": [
            {"op": "activity.create", "local_id": "activity", "kind": "analysis", "title": "Run", "objective": "Score", "source": {"kind": "test", "location": "fixture"}},
            {"op": "frontier.lane.create", "local_id": "lane", "name": "research"},
            {"op": "frontier.item.create", "local_id": "direction", "lane_id": "$lane", "direction": "Score", "rationale": "Test"},
            {"op": "agenda.item.create", "local_id": "agenda", "objective": "score", "frontier_item_ids": ["$direction"]},
            {"op": "execution.outcome.record", "local_id": "one", "source_activity_id": "$activity", "agenda_item_id": "$agenda", "attempt_key": "duplicate", "execution_status": "failed", "artifact_status": "partial", "evaluation_status": "unknown", "audit_status": "not_run", "summary": "first", "evidence": [], "resources": []},
            {"op": "execution.outcome.record", "local_id": "two", "source_activity_id": "$activity", "agenda_item_id": "$agenda", "attempt_key": "duplicate", "execution_status": "succeeded", "artifact_status": "complete", "evaluation_status": "unknown", "audit_status": "not_run", "summary": "duplicate", "evidence": [], "resources": []},
        ],
    })
    from aimx.research.store import database_path
    with pytest.raises(ValidationError) as error:
        apply_update(research_repo, update)
    assert error.value.code == "attempt_conflict"
    assert read_state(research_repo).revision == 0
    assert not database_path(research_repo).exists()


def test_corrupt_event_payload_is_reported_as_unreadable_store(
    research_repo: Path, research_update_factory
) -> None:
    apply_update(research_repo, _update(research_update_factory()))
    with sqlite3.connect(database_path(research_repo)) as connection:
        connection.execute("UPDATE events SET payload = '{'")
        connection.commit()

    with pytest.raises(StoreError) as error:
        read_state(research_repo)

    assert error.value.code == "research_store_unreadable"
