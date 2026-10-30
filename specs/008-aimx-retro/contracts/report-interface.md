# Report helper interface (planned)

This interface belongs to the installed skill, not the Aimx CLI. Python 3.10–3.12 with standard library only; development uses Python 3.12 via uv. No Aim imports, subprocess calls, source-data writes or network access.

```bash
uv run python "$RETRO_SKILL_DIR/scripts/report.py" validate --file "$DRAFT_JSON"
uv run python "$RETRO_SKILL_DIR/scripts/report.py" publish --file "$DRAFT_JSON"
uv run python "$RETRO_SKILL_DIR/scripts/report.py" publish --file "$DRAFT_JSON" --output-root "$REPORT_ROOT"
```

## Input

The version-1 report object is defined in [../data-model.md](../data-model.md). Required fields and all reference rules are normative. Reject duplicate JSON keys, non-finite numbers, unknown fields, wrong types (including booleans in integer fields), unsupported schema versions, duplicate IDs, unresolved references, unsupported verified claims and over-limit reflection/option arrays. Time fields require parseable timezone-bearing timestamps; an end time before its start is invalid. Missing values use explicit null only where the model permits it.

Unknown-field validation catches drafting errors; no speculative extension bag is provided. The model contract will be copied into the skill's report-format reference during implementation so the installed skill is self-contained. The helper validates structure and provenance links, not scientific correctness, causal proof, freshness or actual source availability. `availability` records what the reviewing Agent observed; the helper does not dereference locators.

## Validate

Read the draft, validate completely, return a bounded JSON receipt. Do not create directories, reports or source snapshots. Valid input exits 0. Invalid input exits 2; error paths identify the offending field without dumping input contents.

Success: `{ "ok": true, "schema_version": 1, "report_id": "..." }`.
Failure: `{ "ok": false, "error": { "code": "invalid_report", "message": "...", "path": "next_step.options[0]" } }`.

CLI parsing errors also exit 2; argparse usage may go to stderr instead of a JSON envelope. I/O errors use `io_error`; publication collisions use `output_exists`; invalid destinations use `unsafe_output`.

## Publish

Validate first. Default output root is `<scope.repo>/.aimx/retro`. An explicit root is a dedicated report-output location, never `.aim`, `.aimx/research`, `.aimx/autoresearch` or a symlink resolving inside these protected areas. Destination is `<output-root>/<report_id>/`; refuse any existing destination, including empty directories and symlinks. Resolve destination containment before writes and reject report IDs with traversal or separators. Do not create or migrate a research database.

Produce exactly the authoritative `retro.json` and derived `RETRO.md` report pair. Existing source evidence remains where it was; the Agent may separately preserve relevant command responses in a fresh evidence directory under the same allowed output root. Rendering includes every analytical field, source selector, reference, unknown, exception and limitation, in stable order with readable labels. JSON values must not be interpreted as Markdown directives or executable instructions. No independent LLM rewrite of Markdown after rendering.

Use an exclusively created destination directory. Prepare both file contents before writing; create files exclusively. If a normal write fails, remove only files/directory created by this invocation, never a pre-existing path. After abrupt termination, a partial directory may remain; later calls refuse it and use a new report ID. A report is complete only when both files exist and a successful receipt was returned or both artifacts were verified. Do not claim cross-filesystem transactions or overwrite recovery.

Success exits 0: `{ "ok": true, "report_id": "...", "report_dir": "/absolute/...", "json_path": "/absolute/.../retro.json", "markdown_path": "/absolute/.../RETRO.md" }`.

Repeated publication of the same ID fails safely even for identical input; a new review gets a new ID. Permission failures never trigger alternative writes to protected state. The user can select a writable explicit report root.

## Semantic acceptance rubric

Structural validation is necessary but insufficient. Review source support for every substantive observation; separate historical availability from awareness, execution failure from hypothesis failure, and later success from earlier rationality. Confirm reflections and choices serve the original objective and actual constraints. Reject a suggested experiment with neither result-dependent decisions nor independent value. Check that the recommendation does not imply resuming paused work or executing without Lead/user authorization.

Evaluate complete rounds, checkpoint-free history and incomplete evidence in fresh Agent contexts. Include focused cases for confounding, execution failure, hindsight, unknown awareness, ambiguity, drift, sound process, satisfied goals and paused work. Store scenario inputs and a rubric; accept equivalent sound wording rather than matching a golden paragraph. Preserve evaluation output and review notes as evidence of skill quality.
