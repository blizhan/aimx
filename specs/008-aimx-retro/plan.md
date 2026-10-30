# Implementation Plan: Aimx Research Retro Skill

**Branch**: `codex/008-aimx-retro` | **Date**: 2026-10-29 | **Spec**: [spec.md](spec.md)
**Input**: `/Users/blizhan/data/code/github/aimx/specs/008-aimx-retro/spec.md`

## Summary

Deliver an independent `aimx-retro` skill that helps a Lead choose the next research step from persisted Aimx evidence. Support explicit standalone invocation by a fresh Agent and optional between-round use through the same protocol. Reuse current public read commands and optional saved loop artifacts; distinguish present state from incomplete historical knowledge. Agent authors a structured report; a small standalone helper validates references and generates a matching Markdown report into a fresh directory. No new Aimx command, database entity, scheduler or model dependency.

## Technical Context

**Language/Version**: Markdown skill instructions; Python 3.12 development, helper compatibility >=3.10,<3.13  
**Primary Dependencies**: Python standard library only for report validation/rendering; existing Aimx read CLI and Aim SDK environment for evidence; existing pytest for tests; no new dependencies  
**Storage**: New JSON/Markdown artifacts in `<repo>/.aimx/retro/<report-id>/` or an explicit independent output root; source Aim, Research State and checkpoints read-only  
**Testing**: pytest contract/unit and isolated skill integration tests; fresh-Agent scenario evaluation with evidence-based rubric  
**Target Platform**: Local/SSH terminal environments supporting existing Aimx; standalone report helper works without importing Aim/Aimx  
**Project Type**: Agent skill with a bounded artifact helper in a companion CLI repository  
**Performance Goals**: Select evidence by goal and key decisions; cap reflections/options at three each. No model-latency SLA or blanket full-history scan. Report rendering proportional to supplied draft size.  
**Constraints**: Preserve original goals, paused/terminal state and budgets; unknown historical context/cost remains unknown; no source mutations, new runtime dependency or historical replay API  
**Scale/Scope**: One repository and selected research objective per invocation, possibly multiple historical rounds; two invocation paths, one versioned report contract

## Constitution Check

*Pre-research and post-design review: PASS. No exceptions required.*

- [x] Safe coexistence: no installed Aim changes, executable replacement, monkey patching or `.aim` writes. Only existing read surfaces are used.
- [x] Ownership boundary: skill helper owns validation/publication only. No dispatch/parser changes; all native passthrough remains unchanged.
- [x] Read-only default: source state and artifacts stay read-only. Explicit invocation authorizes only independent report creation. Publication rejects collisions and protected source destinations.
- [x] CLI-first contract: helper has deterministic validation, JSON receipts and explicit failure codes; report pair works over shell/SSH without a GUI. Agent semantics are external to Aimx.
- [x] Compatibility plan: use existing lockfile-resolved Aim environment (declared dev range `aim>=3.29.1`), record actual version in integration evidence, and retain existing passthrough tests. Test script in isolation and verify current read commands on repo-scoped data. No new Aim-version support claim.

## Project Structure

### Documentation (this feature)

```text
specs/008-aimx-retro/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── checklists/requirements.md
└── contracts/
    ├── skill-interface.md
    └── report-interface.md
```

`tasks.md` is intentionally left for `/speckit.tasks`.

### Source Code (repository root; planned changes)

```text
skills/aimx-retro/
├── SKILL.md
├── agents/openai.yaml
├── references/
│   ├── evidence-guide.md
│   └── report-format.md
└── scripts/report.py
skills/aimx-autoresearch/SKILL.md              # conditional handoff pointer
skills/aimx-autoresearch/references/research-handoff.md
README.md
README.zh-CN.md
tests/
├── unit/test_retro_report.py
├── integration/test_retro_skill.py
└── fixtures/retro/
    ├── reports/                             # valid/invalid structured drafts
    └── scenarios/                           # evidence packets and review rubric
```

**Structure Decision**: Follow independently installable existing skill packages. Keep one report model and one deterministic renderer; no collector, runtime abstraction, package dependency or `src/aimx` edit. Evidence guide is loaded when collecting sources; report format when authoring/validating. The primary skill carries ordered analysis and completion criteria. Both README skill lists register discovery and two entry examples.

## Phase 0: Research completed

[research.md](research.md) resolves evidence reads, history limitations, report storage/consistency, optional integration, packaging and validation. Two Luna workers collected repository facts. Existing state/context cannot perform historical revision reads; no new replay feature is planned. Optional checkpoints are references, not complete trajectories or proof of claims.

## Phase 1: Design completed

- [data-model.md](data-model.md): versioned report, scope, evidence, decisions, knowledge updates, reflections and next-step options; explicit unknowns and reference validation.
- [contracts/skill-interface.md](contracts/skill-interface.md): two entries, ordered process, supported read examples, bounded collection and Lead handoff.
- [contracts/report-interface.md](contracts/report-interface.md): validation/publication, errors, destination ownership and semantic rubric.
- [quickstart.md](quickstart.md): standalone and round usage, proposed helper commands and verification workflow.
- Update repository Agent context through the prescribed script, retaining manual instructions and documenting only this feature's technology/storage additions.

## Phase 2: Implementation sequence for later task generation

1. Establish the standalone skill protocol and report-format/evidence references. Cover scope clarification, absent checkpoint/history, evidence provenance, execution/comparison checks, alternative causes, and action branches.
2. Implement report validation and canonical rendering/publication in one stdlib helper. Add substantive tests for malformed input, reference resolution, contract limits, source immutability and collision/failure behavior.
3. Add optional autoresearch handoff and both README entries. Preserve existing checkpoint and ResearchUpdate contracts and make sibling-skill absence harmless.
4. Add isolated installed-skill invocation and semantic scenario packets. Run fresh-Agent reviews for both invocation paths and evaluate against the rubric before considering the skill complete.

Dependencies: report model precedes helper; handoff uses the final report contract; semantic evaluation follows actual skill assembly. No implementation is performed by this planning command.

## Validation strategy

- Run focused unit tests for the report helper and integration tests that copy only `skills/aimx-retro` and invoke under `python -I`; no dependency on repository imports.
- Reject malformed/duplicate-key input, bool-as-integer, non-finite values, unknown version/fields, missing fields, broken references, unsupported observed/tested statuses, invalid time ordering, over-limit arrays and experiment options lacking meaningful result branches or independent value.
- Verify deterministic output and full analytical-field preservation in Markdown; never compare only a few report titles. Check absent/null values render visibly.
- Hash sentinel source files before/after success, validation failure, collision and write failure. Check forbidden destinations, path traversal and symlink redirection, existing reports, and incomplete-publish recovery behavior.
- Exercise real read commands against `/Users/blizhan/data/code/github/aimx/data/.aim` if usable; do not initialize, repair or write it. Use isolated synthetic fixtures for missing/empty/drifting state. Record exact Aim version and evidence coverage. Existing native passthrough tests remain regression gates.
- Semantic evaluation covers SC-001–SC-005 in fresh contexts: complete round, standalone history, incomplete evidence, execution failure, confounding, hindsight, unknown awareness, sound process, satisfied goal and paused work. Rubric checks actual evidence entailment and decision usefulness; structural tests alone cannot claim these pass.
- SC-006/SC-007 additionally use deterministic source-preservation and report-consistency tests. Finish with `uv run pytest` when implementation is ready; follow failures with targeted diagnostics rather than expanding scope.

## Risks and limits

Incomplete historical context limits causal diagnosis by design. Independently sampled runs/files are not a transaction even when research revision is stable. The helper verifies report structure, not scientific truth or file freshness. Future changes to the report schema require a new schema version. Reports preserve source locators; deleted external evidence remains a disclosed gap rather than a promise of a full archive.
