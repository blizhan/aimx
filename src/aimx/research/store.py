from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from aimx.research.errors import RevisionConflict, StoreError, ValidationError
from aimx.research.models import ResearchState, ResearchUpdate
from aimx.research.replay import replay

DB_RELATIVE_PATH = Path(".aimx") / "research" / "state.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS commits (
    id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL UNIQUE,
    base_revision INTEGER NOT NULL,
    client_update_id TEXT UNIQUE,
    author_kind TEXT NOT NULL,
    author_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    message TEXT,
    schema_version INTEGER NOT NULL,
    request_fingerprint TEXT,
    result_json TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    commit_id TEXT NOT NULL REFERENCES commits(id),
    revision INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    entity_id TEXT,
    payload TEXT NOT NULL,
    UNIQUE(commit_id, sequence)
);
CREATE INDEX IF NOT EXISTS idx_events_revision ON events(revision, sequence);
"""


@dataclass(frozen=True)
class CommitResult:
    status: str
    revision: int
    commit_id: str | None = None
    client_update_id: str | None = None
    created_ids: dict[str, str] | None = None
    idempotent: bool = False

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": 1,
            "status": self.status,
            "revision": self.revision,
        }
        if self.commit_id is not None:
            result["commit_id"] = self.commit_id
        if self.client_update_id is not None:
            result["client_update_id"] = self.client_update_id
        if self.created_ids is not None:
            result["created_ids"] = dict(self.created_ids)
        if self.idempotent:
            result["idempotent"] = True
        return result


def normalize_repo_root(repo_path: Path) -> Path:
    path = repo_path.expanduser()
    if not path.exists():
        raise StoreError(f"Repository path does not exist: {path}")
    if path.name == ".aim":
        path = path.parent
    if not path.is_dir():
        raise StoreError(f"Repository path is not a directory: {path}")
    return path.resolve()


def database_path(repo_root: Path) -> Path:
    return normalize_repo_root(repo_root) / DB_RELATIVE_PATH


def has_store(repo_root: Path) -> bool:
    return database_path(repo_root).is_file()


def current_revision(repo_root: Path) -> int:
    """Read the metadata revision without creating or mutating the store."""

    path = database_path(repo_root)
    if not path.exists():
        return 0
    try:
        with _connect_read_only(path) as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'current_revision'"
            ).fetchone()
            if row is not None:
                return int(row[0])
            row = connection.execute("SELECT MAX(revision) FROM commits").fetchone()
            return int(row[0] or 0)
    except (sqlite3.Error, ValueError, TypeError) as exc:
        raise StoreError(f"Unable to read research revision at {path}: {exc}") from exc


def read_commits(repo_root: Path) -> list[dict[str, Any]]:
    path = database_path(repo_root)
    if not path.exists():
        return []
    try:
        with _connect_read_only(path) as connection:
            return [dict(row) for row in connection.execute(
                "SELECT id, revision, base_revision, client_update_id, author_kind, "
                "author_name, created_at, message, schema_version FROM commits "
                "ORDER BY revision, id"
            )]
    except sqlite3.Error as exc:
        raise StoreError(f"Unable to read research commits at {path}: {exc}") from exc


def read_events(repo_root: Path) -> list[dict[str, Any]]:
    path = database_path(repo_root)
    if not path.exists():
        return []
    try:
        with _connect_read_only(path) as connection:
            events: list[dict[str, Any]] = []
            for row in connection.execute(
                "SELECT id, commit_id, revision, sequence, event_type, entity_id, payload "
                "FROM events ORDER BY revision, sequence"
            ):
                value = dict(row)
                value["payload"] = json.loads(value["payload"])
                events.append(value)
            return events
    except (sqlite3.Error, json.JSONDecodeError, TypeError) as exc:
        raise StoreError(f"Unable to read research events at {path}: {exc}") from exc


def read_state(repo_root: Path) -> ResearchState:
    path = database_path(repo_root)
    if not path.exists():
        return ResearchState()
    try:
        with _connect_read_only(path) as connection:
            return _read_connection_state_checked(connection, path, "read")
    except sqlite3.Error as exc:
        raise StoreError(f"Unable to read research state at {path}: {exc}") from exc


def dry_run_update(
    repo_root: Path,
    update: ResearchUpdate,
    evidence_resolver: Callable[[str], str] | None = None,
) -> tuple[ResearchState, dict[str, str]]:
    from aimx.research.validation import compile_update

    path = database_path(repo_root)
    if path.exists():
        try:
            with _connect_read_only(path) as connection:
                state = _read_connection_state_checked(connection, path, "inspect")
        except sqlite3.Error as exc:
            raise StoreError(f"Unable to inspect research state at {path}: {exc}") from exc
    else:
        state = ResearchState()
    try:
        if update.base_revision != state.revision:
            raise RevisionConflict(update.base_revision, state.revision)
        compiled = compile_update(
            update,
            state,
            revision=state.revision + 1,
            commit_id="commit_dry_run",
            evidence_resolver=evidence_resolver,
        )
        return state, compiled.created_ids
    except sqlite3.Error as exc:
        raise StoreError(f"Unable to inspect research state at {path}: {exc}") from exc


def apply_update(
    repo_root: Path,
    update: ResearchUpdate,
    evidence_resolver: Callable[[str], str] | None = None,
) -> CommitResult:
    from aimx.research.ids import new_id
    from aimx.research.validation import compile_update

    root = normalize_repo_root(repo_root)
    path = root / DB_RELATIVE_PATH
    request_fingerprint = _request_fingerprint(update)

    # Validate against a read-only snapshot before creating the sidecar. This
    # keeps malformed or stale writes from leaving an empty database behind.
    if path.exists():
        try:
            with _connect_read_only(path) as connection:
                snapshot = _read_connection_state_checked(connection, path, "inspect")
                existing = _existing_client_commit(connection, update.client_update_id)
        except sqlite3.Error as exc:
            raise StoreError(f"Unable to inspect research state at {path}: {exc}") from exc
    else:
        snapshot = ResearchState()
        existing = None
    if existing is not None:
        return _idempotent_result(existing, request_fingerprint, update.client_update_id)
    if update.base_revision != snapshot.revision:
        raise RevisionConflict(update.base_revision, snapshot.revision)
    compile_update(
        update,
        snapshot,
        revision=snapshot.revision + 1,
        commit_id="commit_validation",
        evidence_resolver=evidence_resolver,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _connect_write(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = _existing_client_commit(connection, update.client_update_id)
            if existing is not None:
                connection.rollback()
                return _idempotent_result(
                    existing, request_fingerprint, update.client_update_id
                )

            state = _read_connection_state_checked(connection, path, "read")
            current_revision = state.revision
            if update.base_revision != current_revision:
                raise RevisionConflict(update.base_revision, current_revision)

            commit_id = new_id("commit")
            next_revision = current_revision + 1
            compiled = compile_update(
                update,
                state,
                revision=next_revision,
                commit_id=commit_id,
                evidence_resolver=evidence_resolver,
            )
            created_at = _utc_now()
            committed_result = CommitResult(
                status="committed",
                revision=next_revision,
                commit_id=commit_id,
                client_update_id=update.client_update_id,
                created_ids=compiled.created_ids,
            )
            connection.execute(
                """INSERT INTO commits
                   (id, revision, base_revision, client_update_id, author_kind,
                    author_name, created_at, message, schema_version,
                    request_fingerprint, result_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    commit_id,
                    next_revision,
                    update.base_revision,
                    update.client_update_id,
                    update.author.kind,
                    update.author.name,
                    created_at,
                    update.message,
                    update.schema_version,
                    request_fingerprint,
                    json.dumps(
                        committed_result.as_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
            for sequence, event in enumerate(compiled.events):
                connection.execute(
                    """INSERT INTO events
                       (id, commit_id, revision, sequence, event_type, entity_id, payload)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        new_id("event"),
                        commit_id,
                        next_revision,
                        sequence,
                        event["event_type"],
                        event.get("entity_id"),
                        json.dumps(event["payload"], ensure_ascii=False, sort_keys=True),
                    ),
                )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', '1')"
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('current_revision', ?)",
                (str(next_revision),),
            )
            connection.commit()
            return committed_result
    except (sqlite3.IntegrityError, json.JSONDecodeError, TypeError, KeyError) as exc:
        raise StoreError(f"Research state write was rejected: {exc}") from exc
    except sqlite3.Error as exc:
        raise StoreError(f"Unable to write research state at {path}: {exc}") from exc


def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _connect_write(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(_SCHEMA)
    _ensure_commit_idempotency_columns(connection)
    return connection


def _read_connection_state(connection: sqlite3.Connection) -> ResearchState:
    commits = []
    for row in connection.execute(
        "SELECT id, revision, base_revision, client_update_id, author_kind, "
        "author_name, created_at, message, schema_version FROM commits "
        "ORDER BY revision, id"
    ):
        commits.append(dict(row))

    events = []
    for row in connection.execute(
        "SELECT id, commit_id, revision, sequence, event_type, entity_id, payload "
        "FROM events ORDER BY revision, sequence"
    ):
        value = dict(row)
        value["payload"] = json.loads(value["payload"])
        events.append(value)
    return replay(commits, events)


def _existing_client_commit(
    connection: sqlite3.Connection, client_update_id: str | None
) -> dict[str, Any] | None:
    if not client_update_id:
        return None
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(commits)")
    }
    request_fingerprint = (
        "request_fingerprint" if "request_fingerprint" in columns else "NULL AS request_fingerprint"
    )
    result_json = "result_json" if "result_json" in columns else "NULL AS result_json"
    row = connection.execute(
        f"SELECT id, revision, {request_fingerprint}, {result_json} "
        "FROM commits WHERE client_update_id = ?",
        (client_update_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def _request_fingerprint(update: ResearchUpdate) -> str:
    payload = json.dumps(
        update.as_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _idempotent_result(
    existing: dict[str, Any],
    request_fingerprint: str,
    client_update_id: str | None,
) -> CommitResult:
    if existing.get("request_fingerprint") != request_fingerprint:
        raise ValidationError(
            "client_update_id is already bound to a different ResearchUpdate.",
            code="idempotency_conflict",
            client_update_id=client_update_id,
        )
    raw_result = existing.get("result_json")
    if not isinstance(raw_result, str):
        raise StoreError(
            "Stored idempotency result is unavailable for this client_update_id.",
            client_update_id=client_update_id,
        )
    try:
        payload = json.loads(raw_result)
        created_ids = payload.get("created_ids")
        if created_ids is not None and not isinstance(created_ids, dict):
            raise TypeError("created_ids must be an object")
        return CommitResult(
            status=str(payload["status"]),
            revision=int(payload["revision"]),
            commit_id=str(payload["commit_id"]),
            client_update_id=payload.get("client_update_id"),
            created_ids=created_ids,
            idempotent=True,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise StoreError(
            f"Stored idempotency result is unreadable: {exc}",
            client_update_id=client_update_id,
        ) from exc


def _ensure_commit_idempotency_columns(connection: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(commits)")
    }
    changed = False
    if "request_fingerprint" not in columns:
        connection.execute("ALTER TABLE commits ADD COLUMN request_fingerprint TEXT")
        changed = True
    if "result_json" not in columns:
        connection.execute("ALTER TABLE commits ADD COLUMN result_json TEXT")
        changed = True
    if changed:
        connection.commit()


def _read_connection_state_checked(
    connection: sqlite3.Connection, path: Path, action: str
) -> ResearchState:
    try:
        return _read_connection_state(connection)
    except (sqlite3.Error, json.JSONDecodeError, TypeError, KeyError, ValueError, ValidationError) as exc:
        raise StoreError(f"Unable to {action} research state at {path}: {exc}") from exc


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
