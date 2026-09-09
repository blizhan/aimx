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
