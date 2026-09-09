from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aimx.commands.finding import run_finding_command
from aimx.commands.lineage import run_lineage_command
from aimx.commands.research import run_research_command
from aimx.research.models import ResearchUpdate
from aimx.research.store import apply_update, read_state


def _json_output(result) -> dict:
    assert result.output is not None
    return json.loads(result.output)


def test_findings_lineage_and_annotations_survive_a_second_process(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    aim_dir = repo / ".aim"
    aim_dir.mkdir()
    sentinel = aim_dir / "sentinel"
    sentinel.write_text("aim-data", encoding="utf-8")
    before = hashlib.sha256(sentinel.read_bytes()).hexdigest()

    update = {
        "schema_version": 1,
        "base_revision": 0,
        "author": {"kind": "agent", "name": "round-one"},
        "operations": [
            {
                "op": "finding.create",
                "local_id": "f1",
                "claim": "The candidate improves accuracy on low data.",
            },
            {
                "op": "finding.create",
                "local_id": "f2",
                "claim": "The improvement needs a second seed.",
            },
            {
                "op": "relation.create",
                "local_id": "r1",
                "source": "$f2",
                "target": "$f1",
                "relation_type": "tests",
            },
        ],
    }
    update_file = tmp_path / "update.json"
    update_file.write_text(json.dumps(update), encoding="utf-8")
    first = run_research_command(
        ["update", "--repo", str(repo), "--file", str(update_file), "--json"]
    )
    first_payload = _json_output(first)
    finding_id = first_payload["created_ids"]["f1"]
    second_finding_id = first_payload["created_ids"]["f2"]
    relation_id = first_payload["created_ids"]["r1"]

    assert run_finding_command(
        ["comment", finding_id, "review", "the", "seed", "--repo", str(repo), "--json"]
    ).exit_status == 0
    assert run_finding_command(
        ["accept", finding_id, "--reason", "Useful working result", "--repo", str(repo), "--json"]
    ).exit_status == 0
    assert run_finding_command(
        [
            "assess",
            finding_id,
            "--status",
            "validated",
            "--confidence",
            "high",
            "--repo",
            str(repo),
            "--json",
        ]
    ).exit_status == 0

    finding_list = _json_output(run_finding_command(["ls", "--repo", str(repo), "--json"]))
    assert {item["id"] for item in finding_list["findings"]} == {finding_id, second_finding_id}
    assert next(item for item in finding_list["findings"] if item["id"] == finding_id)[
        "governance_status"
    ] == "accepted"

    lineage = _json_output(
        run_lineage_command(["show", finding_id, "--repo", str(repo), "--json"])
    )
    assert relation_id in {item["id"] for item in lineage["relations"]}

    after = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    assert before == after


def test_finding_shorthand_rejects_unknown_options_without_committing(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    finding_id = _commit_finding_with_evidence(repo, "a" * 32)

    result = run_finding_command(
        ["accept", finding_id, "--reasn", "typo", "--repo", str(repo), "--json"]
    )

    assert result.exit_status == 2
    assert read_state(repo).revision == 1


def test_research_context_reads_durable_annotations(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    update_file = tmp_path / "update.json"
    update_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_revision": 0,
                "author": {"kind": "agent", "name": "agent"},
                "operations": [
                    {"op": "finding.create", "local_id": "f1", "claim": "accuracy result"},
                    {
                        "op": "annotation.create",
                        "annotation_target_kind": "finding",
                        "annotation_target_id": "$f1",
                        "text": "accuracy needs a controlled follow-up",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    committed = _json_output(
        run_research_command(
            ["update", "--repo", str(repo), "--file", str(update_file), "--json"]
        )
    )
    assert committed["revision"] == 1
    context = _json_output(
        run_research_command(
            [
                "context",
                "--repo",
                str(repo),
                "--objective",
                "accuracy follow-up",
                "--budget",
                "10000",
                "--json",
            ]
        )
    )
    assert context["revision"] == 1
    assert any(item["kind"] == "annotation" for item in context["items"])


def _commit_finding_with_evidence(repo: Path, run_hash: str) -> str:
    update = ResearchUpdate.from_dict(
        {
            "schema_version": 1,
            "base_revision": 0,
            "author": {"kind": "agent", "name": "agent"},
            "operations": [
                {
                    "op": "finding.create",
                    "local_id": "f1",
                    "claim": "Evidence-backed finding.",
                    "evidence": [{"kind": "aim_run", "run_hash": run_hash}],
                }
            ],
        }
    )
    result = apply_update(repo, update, evidence_resolver=lambda value: value)
    assert result.created_ids is not None
    return result.created_ids["f1"]


def test_finding_read_marks_missing_aim_evidence_unavailable(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_hash = "a" * 32
    finding_id = _commit_finding_with_evidence(repo, run_hash)

    payload = _json_output(
        run_finding_command(["show", finding_id, "--repo", str(repo), "--json"])
    )

    assert payload["finding"]["evidence"] == [
        {"kind": "aim_run", "run_hash": run_hash, "availability": "unavailable"}
    ]
    human = run_finding_command(["show", finding_id, "--repo", str(repo)])
    assert human.output is not None
    assert f"{run_hash} (unavailable)" in human.output


def test_finding_read_marks_resolvable_aim_evidence_available(
    tmp_path: Path, monkeypatch
) -> None:
    import aimx.aim_bridge.research_evidence as research_evidence

    repo = tmp_path / "repo"
    repo.mkdir()
    run_hash = "b" * 32
    finding_id = _commit_finding_with_evidence(repo, run_hash)

    class _ReadableRepo:
        def __init__(self, path: str, *, read_only: bool | None = None) -> None:
            assert path == str(repo)
            assert read_only is True

        def list_all_runs(self):
            return [run_hash]

    monkeypatch.setattr(research_evidence, "Repo", _ReadableRepo)

    payload = _json_output(
        run_finding_command(["show", finding_id, "--repo", str(repo), "--json"])
    )

    assert payload["finding"]["evidence"] == [
        {"kind": "aim_run", "run_hash": run_hash, "availability": "available"}
    ]
