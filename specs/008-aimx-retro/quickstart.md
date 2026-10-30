# Aimx Retro Quickstart

These instructions describe the delivered `aimx-retro` skill and helper. Use the repository's existing `uv` workflow with Python 3.12 for development; no additional environment or external dependencies are required.

## Standalone retrospective

Install the repository's skills using its documented installation workflow, then invoke the new skill in an Agent host that can read the target repository:

```text
用 $aimx-retro 复盘 /absolute/path/to/research-repo 中关于模型泛化的研究。
重点检查已有结论是否可靠，以及下一步最值得解决的问题。
没有之前的聊天记录；没有 autoresearch checkpoint。
```

Optional refinements:

```text
只分析第 2–5 轮；重点比较继续调参与做特征工程的依据。
loop 记录位于 /absolute/path/to/research-repo/.aimx/autoresearch/study-a。
```

The Agent establishes scope, reads current state and selected evidence, reconstructs only supported history, and produces a bounded next-step review. Multiple possible objectives prompt a selection; missing records limit analysis rather than require a checkpoint. Empty evidence can produce a useful missing-evidence report. No experiment or research update starts as a side effect.

## Between rounds

After results, integration and aggregate ResearchUpdate are reconciled, the Lead may ask:

```text
使用 $aimx-retro 复盘刚完成的这一轮和相关历史。
仓库：/absolute/path/to/research-repo
目标：在既定数据划分下确认候选方法是否值得继续投入。
提供本轮 task/result 文件、研究 revision 和剩余约束。
给出本轮认知更新及下一轮可比较选项；保持现有研究状态不变。
```

The Lead reads the report, checks current revision, user steering and constraints, then decides whether to adopt advice through existing workflows. A stale report requires rechecking affected claims. A paused or terminal loop is not resumed by either reviewing or publishing a report.

## Report creation

The Agent authors a version-1 draft following the installed skill's report-format reference. Resolve `RETRO_SKILL_DIR` to its absolute installed directory and `DRAFT_JSON` to the separate draft file. These are proposed helper calls, not new Aimx commands:

```bash
uv run python "$RETRO_SKILL_DIR/scripts/report.py" validate --file "$DRAFT_JSON"
uv run python "$RETRO_SKILL_DIR/scripts/report.py" publish --file "$DRAFT_JSON"
```

Default result: `/absolute/repo/.aimx/retro/<report-id>/retro.json` and `RETRO.md`. For a read-only repository, supply `--output-root /absolute/writable/retro-reports`. Existing IDs are never overwritten; create a new ID for another retrospective. Optional copied evidence responses belong in a fresh independent evidence directory, outside source data and checkpoints.

Read the scope and gaps first, then knowledge updates, reflections and next-step options. Zero process problems is a valid result. A recommended option explains the uncertainty it resolves, prerequisites, known/unknown cost and how different outcomes change the subsequent decision. Report publication confirms format validity, not scientific correctness.

## Developer validation after implementation

From `/Users/blizhan/data/code/github/aimx`:

```bash
uv run pytest tests/unit/test_retro_report.py tests/integration/test_retro_skill.py
uv run aimx research state --repo /Users/blizhan/data/code/github/aimx/data/.aim --json
uv run aim runs --repo /Users/blizhan/data/code/github/aimx/data/.aim ls
uv run pytest
```

Use the local Aim repository only if available; no initialization or repair. The feature-specific tests are planned, not present at planning time. Evaluate saved scenario packets in fresh Agent contexts for complete rounds, standalone history and insufficient evidence. Review outputs against the semantic rubric and preserve review notes. Deterministic tests do not validate reasoning quality.

## Continue Spec Kit

Actual branch remains `codex/008-aimx-retro`. This installation's scripts require a leading-number feature identity, so use the supported override for scripts that check it:

```bash
SPECIFY_FEATURE=008-aimx-retro .specify/scripts/bash/check-prerequisites.sh --json
```

The persisted feature directory is `specs/008-aimx-retro`. Next run `/speckit.tasks` to generate implementation tasks from this plan; do not rerun setup-plan over the completed plan.
