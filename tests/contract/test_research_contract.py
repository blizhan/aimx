from __future__ import annotations

import json
from pathlib import Path

import pytest

from aimx.commands.research import run_research_command
from aimx.research.context import compile_context
from aimx.research.models import ResearchState, json_bytes


def _write_update(path: Path, *, base_revision: int = 0, claim: str = "accuracy claim") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_revision": base_revision,
                "client_update_id": f"update-{base_revision}-{claim}",
                "author": {"kind": "agent", "name": "contract-agent"},
                "operations": [
                    {
                        "op": "finding.create",
                        "local_id": "f1",
                        "claim": claim,
                        "epistemic_status": "candidate",
                        "confidence": "medium",
                        "governance_status": "proposed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_research_state_empty_and_update_contracts_are_json_stable(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    empty = run_research_command(["state", "--repo", str(repo), "--json"])
    empty_payload = json.loads(empty.output)
    assert empty.exit_status == 0
    assert empty_payload["schema_version"] == 1
    assert empty_payload["revision"] == 0
    assert empty_payload["findings"] == []
    assert not (repo / ".aimx").exists()

    update_file = tmp_path / "update.json"
    _write_update(update_file)
    dry_run = run_research_command(
        ["update", "--repo", str(repo), "--file", str(update_file), "--dry-run", "--json"]
    )
    dry_payload = json.loads(dry_run.output)
    assert dry_run.exit_status == 0
    assert dry_payload == {
        "base_revision": 0,
        "schema_version": 1,
        "status": "valid",
        "would_create": ["finding"],
    }
    assert not (repo / ".aimx").exists()

    committed = run_research_command(
        ["update", "--repo", str(repo), "--file", str(update_file), "--json"]
    )
    committed_payload = json.loads(committed.output)
    assert committed.exit_status == 0
    assert committed_payload["status"] == "committed"
    assert committed_payload["revision"] == 1
    assert committed_payload["created_ids"]["f1"].startswith("f_")


def test_complete_context_response_fits_the_declared_utf8_budget() -> None:
    state = ResearchState(revision=7)
    payload = compile_context(state, "候选", 4096)
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    assert len(encoded) <= payload["budget_limit"]
    assert payload["budget_scope"] == "response"
    assert payload["budget_used"] == len(encoded)


def test_context_budget_exact_boundary_and_one_byte_under() -> None:
    from aimx.research.errors import BudgetError

    state = ResearchState(findings={"f1": {"id": "f1", "claim": "accuracy improves", "evidence": [], "epistemic_status": "candidate", "confidence": "medium", "governance_status": "proposed", "created_revision": 1, "created_commit_id": "c1", "author": {"kind": "agent", "name": "a"}}})
    initial = compile_context(state, "accuracy", 10_000)
    exact = compile_context(state, "accuracy", initial["budget_used"])["budget_used"]
    assert compile_context(state, "accuracy", exact)["budget_used"] == exact
    with pytest.raises(BudgetError) as error:
        compile_context(state, "accuracy", exact - 1)
    assert error.value.details["required_bytes"] == exact
    assert error.value.details["query_commands"]


def test_context_cli_budget_error_has_empty_stdout_and_json_scope() -> None:
    result = run_research_command(["context", "--objective", "accuracy", "--budget", "1", "--json"])
    assert result.exit_status == 2
    assert result.output is None
    payload = json.loads(result.error_message)
    assert payload["error"]["code"] == "budget_too_small_for_required_context"
    assert payload["error"]["details"]["query_commands"]


def test_budget_too_small_reports_a_complete_required_size() -> None:
    from aimx.research.errors import BudgetError

    with pytest.raises(BudgetError) as error:
        compile_context(ResearchState(), "候选", 1)
    assert error.value.code == "budget_too_small_for_required_context"
    assert error.value.details["required_bytes"] >= json_bytes({})
    assert error.value.details["query_commands"]


def test_context_human_output_discloses_complete_json_budget_scope() -> None:
    result = run_research_command(["context", "--objective", "accuracy", "--budget", "4096"])
    assert result.exit_status == 0
    assert "budget scope: response_json" in result.output
    assert "UTF-8 bytes" in result.output


def test_budget_error_details_are_written_only_to_stderr(capsys) -> None:
    from aimx.__main__ import main

    exit_code = main(["research", "context", "--objective", "accuracy", "--budget", "1", "--json"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "budget_too_small_for_required_context"
    assert payload["error"]["details"]["required_bytes"] > payload["error"]["details"]["budget_limit"]
    assert payload["error"]["details"]["query_commands"]


def test_invalid_and_stale_updates_have_expected_exit_status_and_error_shape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    update_file = tmp_path / "update.json"
    _write_update(update_file)
    assert run_research_command(
        ["update", "--repo", str(repo), "--file", str(update_file), "--json"]
    ).exit_status == 0

    stale_file = tmp_path / "stale.json"
    _write_update(stale_file, base_revision=0, claim="stale claim")
    stale = run_research_command(
        ["update", "--repo", str(repo), "--file", str(stale_file), "--json"]
    )
    stale_payload = json.loads(stale.error_message)
    assert stale.exit_status == 3
    assert stale_payload["error"]["code"] == "revision_conflict"
    assert stale_payload["error"]["details"] == {"base_revision": 0, "current_revision": 1}

    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("{\"schema_version\": 1}", encoding="utf-8")
    invalid = run_research_command(
        ["update", "--repo", str(repo), "--file", str(invalid_file), "--json"]
    )
    invalid_payload = json.loads(invalid.error_message)
    assert invalid.exit_status == 2
    assert invalid_payload["error"]["code"] == "invalid_schema"
