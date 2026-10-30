from __future__ import annotations

import json
import copy
import html
from pathlib import Path
import pytest

# Fixture paths
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "retro" / "reports"
COMPLETE_JSON = FIXTURES_DIR / "complete.json"
MINIMAL_JSON = FIXTURES_DIR / "minimal.json"

# In T005 we will implement skills/aimx-retro/scripts/report.py
# We can import it dynamically or directly:
import sys
SKILL_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "skills" / "aimx-retro" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

import report


def test_fixtures_exist():
    assert COMPLETE_JSON.is_file()
    assert MINIMAL_JSON.is_file()


def test_validate_valid_minimal_and_complete():
    for path in [MINIMAL_JSON, COMPLETE_JSON]:
        data = json.loads(path.read_text(encoding="utf-8"))
        res = report.validate_report_dict(data)
        assert res["ok"] is True
        assert res["report_id"] == data["report_id"]
        assert res["schema_version"] == 1


def test_validate_aim_views_absent_is_backward_compatible():
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    assert "aim_views" not in data
    assert report.validate_report_dict_safe(data)["ok"] is True


def test_validate_rejects_invalid_aim_view_page():
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["aim_views"] = [{
        "id": "bad_page",
        "page": "unknown",
        "purpose": "Check the round runs",
        "query": "run.hash != ''",
        "url": None,
    }]
    result = report.validate_report_dict_safe(data)
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_report"
    assert result["error"]["path"] == "aim_views[0].page"


def test_aim_views_render_with_query_and_fallback_instruction():
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["aim_views"] = [{
        "id": "round_1_runs",
        "page": "runs",
        "purpose": "Compare round 1 roles",
        "query": "run.aimx.loop_id == 'study' and run.aimx.round == 1",
        "group_by": ["run.aimx.role"],
        "color_by": ["run.aimx.role"],
        "evidence_ids": ["ev_01"],
        "decision_ids": ["dec_01"],
        "url": None,
    }]
    report.validate_report_dict(data)
    markdown = html.unescape(report.render_markdown(data))
    assert "## Aim Views" in markdown
    assert "### round_1_runs: Compare round 1 roles" in markdown
    assert "run.aimx.loop_id == 'study' and run.aimx.round == 1" in markdown
    assert "Open the runs page in Aim and paste the query into its search box." in markdown


def test_validate_rejects_duplicate_json_keys(tmp_path):
    bad_json = tmp_path / "dup.json"
    bad_json.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    with pytest.raises(report.ReportValidationError) as excinfo:
        report.load_report_file(bad_json)
    assert excinfo.value.code == "invalid_report"
    assert "duplicate" in excinfo.value.message.lower()


def test_validate_rejects_non_finite_numbers():
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["scope"]["revision_start"] = float("nan")
    res = report.validate_report_dict_safe(data)
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_report"


def test_validate_rejects_boolean_as_integer():
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["schema_version"] = True  # In Python, isinstance(True, int) is True, must reject bool!
    res = report.validate_report_dict_safe(data)
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_report"
    assert "schema_version" in res["error"]["path"]


def test_validate_rejects_unknown_fields():
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["extra_unknown_field"] = "unexpected"
    res = report.validate_report_dict_safe(data)
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_report"
    assert "extra_unknown_field" in res["error"]["message"]


def test_validate_time_ordering():
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["scope"]["read_started_at"] = "2026-10-29T11:00:00Z"
    data["scope"]["read_finished_at"] = "2026-10-29T10:00:00Z"
    res = report.validate_report_dict_safe(data)
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_report"
    assert "read_finished_at" in res["error"]["path"]


def test_validate_unresolved_evidence_id():
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["decisions"][0]["choice"]["evidence_ids"] = ["non_existent_ev"]
    res = report.validate_report_dict_safe(data)
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_report"
    assert "non_existent_ev" in res["error"]["message"]


def test_validate_observed_basis_requires_primary_read():
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    # Change ev_01 to researcher_summary
    data["evidence"][0]["provenance"] = "researcher_summary"
    res = report.validate_report_dict_safe(data)
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_report"
    assert "observed" in res["error"]["message"].lower()


def test_validate_reflection_limit():
    data = json.loads(COMPLETE_JSON.read_text(encoding="utf-8"))
    base_ref = data["reflections"][0]
    data["reflections"] = [
        dict(base_ref, id="ref_01"),
        dict(base_ref, id="ref_02"),
        dict(base_ref, id="ref_03"),
        dict(base_ref, id="ref_04"),
    ]
    res = report.validate_report_dict_safe(data)
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_report"
    assert "reflections" in res["error"]["path"]


def test_validate_experiment_option_branches():
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    # Experiment without independent_value and with only 1 branch
    data["next_step"]["options"][0]["outcome_branches"] = [{"result": "loss < 0.4", "next_decision": "stop"}]
    data["next_step"]["options"][0]["independent_value"] = None
    res = report.validate_report_dict_safe(data)
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_report"
    assert "outcome_branches" in res["error"]["message"].lower() or "independent_value" in res["error"]["message"].lower()


def test_canonical_markdown_rendering():
    data = json.loads(COMPLETE_JSON.read_text(encoding="utf-8"))
    md = html.unescape(report.render_markdown(data))
    assert "# Research Retrospective: retro_complete_01" in md
    assert "## Scope" in md
    assert "Maximize classification accuracy while keeping latency under 50ms" in md
    assert "## Evidence" in md
    assert "ev_01" in md
    assert "## Decisions" in md
    assert "dec_01" in md
    assert "## Knowledge Updates" in md
    assert "## Reflections" in md
    assert "ref_01" in md
    assert "## Next Steps & Recommendation" in md
    assert "opt_01" in md
    assert "## Limitations" in md
    assert "Seed variance unmeasured" in md


def test_publish_creates_pair_and_receipt(tmp_path):
    draft_file = tmp_path / "draft.json"
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["scope"]["repo"] = str(tmp_path)
    draft_file.write_text(json.dumps(data), encoding="utf-8")

    res = report.publish_report(draft_file, output_root=None)
    assert res["ok"] is True
    rep_dir = Path(res["report_dir"])
    assert rep_dir.is_dir()
    assert (rep_dir / "retro.json").is_file()
    assert (rep_dir / "RETRO.md").is_file()

    # Check that retro.json matches data
    published_json = json.loads((rep_dir / "retro.json").read_text(encoding="utf-8"))
    assert published_json["report_id"] == data["report_id"]


def test_publish_rejects_collision(tmp_path):
    draft_file = tmp_path / "draft.json"
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["scope"]["repo"] = str(tmp_path)
    draft_file.write_text(json.dumps(data), encoding="utf-8")

    res1 = report.publish_report(draft_file, output_root=None)
    assert res1["ok"] is True

    # Second attempt with same ID should raise collision error
    with pytest.raises(report.ReportValidationError) as excinfo:
        report.publish_report(draft_file, output_root=None)
    assert excinfo.value.code == "output_exists"


def test_publish_rejects_protected_paths(tmp_path):
    draft_file = tmp_path / "draft.json"
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["scope"]["repo"] = str(tmp_path)
    draft_file.write_text(json.dumps(data), encoding="utf-8")

    # Output root targeting .aim directly
    protected_root = tmp_path / ".aim"
    protected_root.mkdir()
    with pytest.raises(report.ReportValidationError) as excinfo:
        report.publish_report(draft_file, output_root=protected_root)
    assert excinfo.value.code == "unsafe_output"


def test_validate_valid_clarify_and_stop_dispositions():
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["next_step"]["disposition"] = "clarify"
    data["next_step"]["recommended_option_id"] = None
    data["next_step"]["options"] = []
    res = report.validate_report_dict_safe(data)
    assert res["ok"] is True

    data["next_step"]["disposition"] = "stop"
    res2 = report.validate_report_dict_safe(data)
    assert res2["ok"] is True


def test_validate_unknown_rationale_and_cost():
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["decisions"][0]["recorded_rationale"] = {
        "text": "Rationale unknown or unrecorded",
        "basis": "unknown",
        "evidence_ids": [],
    }
    data["next_step"]["options"][0]["cost"] = None
    res = report.validate_report_dict_safe(data)
    assert res["ok"] is True


def test_validate_rejects_missing_evidence_supporting_observed():
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["evidence"][0]["availability"] = "missing"
    res = report.validate_report_dict_safe(data)
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_report"
    assert "observed" in res["error"]["message"].lower()


def test_all_report_fields_are_required():
    original = json.loads(COMPLETE_JSON.read_text(encoding="utf-8"))

    def objects(value, path=()):
        if isinstance(value, dict):
            yield path, value
            for key, item in value.items():
                yield from objects(item, (*path, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from objects(item, (*path, index))

    for path, obj in objects(original):
        for field in obj:
            data = copy.deepcopy(original)
            target = data
            for key in path:
                target = target[key]
            del target[field]
            result = report.validate_report_dict_safe(data)
            assert result["ok"] is False, (*path, field)
            assert result["error"]["code"] == "invalid_report"
            assert result["error"]["path"].endswith(field)


def test_missing_nullable_field_prevents_publication(tmp_path):
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    del data["scope"]["objective"]
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(data))
    output = tmp_path / "reports"
    with pytest.raises(report.ReportValidationError, match="Missing required field"):
        report.publish_report(draft, output)
    assert not output.exists()


def test_markdown_keeps_every_statement_reference():
    data = json.loads(COMPLETE_JSON.read_text(encoding="utf-8"))
    statements = []

    def mark(value):
        if isinstance(value, dict):
            if set(value) == {"text", "basis", "evidence_ids"}:
                value["text"] = f"Unique statement {len(statements)}"
                statements.append(value)
            for item in value.values():
                mark(item)
        elif isinstance(value, list):
            for item in value:
                mark(item)

    mark(data)
    data["reflections"][0]["validation_evidence_ids"] = [data["evidence"][0]["id"]]
    report.validate_report_dict(data)
    md = html.unescape(report.render_markdown(data))
    for statement in statements:
        refs = ", ".join(statement["evidence_ids"]) or "none"
        assert f"[{statement['basis']}] {statement['text']} (ev: {refs})" in md
    assert f"**Validation Evidence**: {data['evidence'][0]['id']}" in md


@pytest.mark.parametrize("payload", [
    "<!-- hide report\n", "\n# Forged heading", "[link](https://example.com)",
    "`code` **bold** _italic_", "::directive{key=\"value\"}",
    "<script>alert(1)</script>", "&#60;!-- comment", "\n    indented code",
])
def test_markdown_treats_source_text_as_literal(payload):
    data = json.loads(COMPLETE_JSON.read_text(encoding="utf-8"))
    data["scope"]["objective"] = payload
    data["evidence"][0]["locator"] = payload
    data["decisions"][0]["choice"]["text"] = payload
    original = copy.deepcopy(data)
    report.validate_report_dict(data)
    md = report.render_markdown(data)
    assert payload not in md
    assert payload.replace("\n", "<br>") in html.unescape(md)
    assert "## Limitations" in md
    assert data == original


@pytest.mark.parametrize("metadata", [".aim", ".aimx/research", ".aimx/autoresearch"])
@pytest.mark.parametrize("symlink", [False, True])
def test_publish_rejects_other_repository_metadata(tmp_path, metadata, symlink):
    repo = tmp_path / "source"
    repo.mkdir()
    protected = tmp_path / "other" / metadata
    protected.mkdir(parents=True)
    output = protected / "reports"
    if symlink:
        alias = tmp_path / "alias"
        alias.symlink_to(protected, target_is_directory=True)
        output = alias / "reports"
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["scope"]["repo"] = str(repo)
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(data))
    with pytest.raises(report.ReportValidationError) as exc:
        report.publish_report(draft, output)
    assert exc.value.code == "unsafe_output"
    assert list(protected.iterdir()) == []


def test_publish_rejects_metadata_repository_root(tmp_path):
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    protected = tmp_path / ".aim"
    protected.mkdir()
    data["scope"]["repo"] = str(protected)
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(data))
    with pytest.raises(report.ReportValidationError) as exc:
        report.publish_report(draft)
    assert exc.value.code == "unsafe_output"
    assert list(protected.iterdir()) == []


def test_publish_rejects_relocated_metadata(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    storage = tmp_path / "storage"
    storage.mkdir()
    (repo / ".aim").symlink_to(storage, target_is_directory=True)
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["scope"]["repo"] = str(repo)
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(data))
    with pytest.raises(report.ReportValidationError) as exc:
        report.publish_report(draft, storage / "reports")
    assert exc.value.code == "unsafe_output"
    assert list(storage.iterdir()) == []


@pytest.mark.parametrize("report_id", ["research", "autoresearch"])
def test_publish_checks_final_destination(tmp_path, report_id):
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["scope"]["repo"] = str(tmp_path / "source")
    data["report_id"] = report_id
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(data))
    output = tmp_path / "other" / ".aimx"
    with pytest.raises(report.ReportValidationError) as exc:
        report.publish_report(draft, output)
    assert exc.value.code == "unsafe_output"
    assert not output.exists()


def test_publish_rejects_other_relocated_metadata(tmp_path):
    storage = tmp_path / "storage"
    storage.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    (other / ".aim").symlink_to(storage, target_is_directory=True)
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["scope"]["repo"] = str(tmp_path / "source")
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(data))
    with pytest.raises(report.ReportValidationError) as exc:
        report.publish_report(draft, other / ".aim" / "reports")
    assert exc.value.code == "unsafe_output"
    assert list(storage.iterdir()) == []
