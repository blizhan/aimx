---
name: aimx-autoresearch
description: Run bounded, multi-round research with Aimx shared state, multiple subagents per round, optional per-agent model and reasoning effort, evidence-based goal checks, and explicit checkpoint resume.
---

# Aimx AutoResearch

Use this skill to lead a finite research loop. Aimx stores research meaning;
the host agent runtime executes tasks. This skill also works when installed
alone: its script uses only Python's standard library, and its handoff reference
contains the required Aimx commands. Require an available `aimx` CLI and follow
the target project's environment workflow (`uv run` in the Aimx repository).

## Ownership: lead writes, subagents report

**The lead is the only Research State and loop-checkpoint writer in this loop.**
It creates Agenda/Frontier, reviews evidence, authors the final ResearchUpdate,
runs its dry-run, commits it, handles conflicts, integrates changes, and judges
the goal. Record the lead's identity in `author.name`.

**Subagents return experiment results and evidence, never a final ResearchUpdate.**
They must not invoke `research update`, writing `finding`/`frontier`/`lineage`
commands, the checkpoint helper, or write shared `.aimx` files directly. They
may read research state, edit their own worktree, and run the project's normal
training/logging workflow. This is a workflow rule, not a CLI role/access gate;
other researchers may still write state, so optimistic revision checks apply.
Include these restrictions in **every** task, using the dispatch template in
[references/loop-contract.md](references/loop-contract.md).

## Prepare the bounded plan

Read [references/loop-contract.md](references/loop-contract.md) for config,
checkpoint, and result formats. Resolve script paths from **this installed
skill directory**, not the target project's `skills/` directory. The examples
below use `AUTORESEARCH_SKILL_DIR` for that absolute directory.

1. Convert natural-language requests into the same JSON config used for file
   invocation. Establish a finite `max_rounds`, agent list, goal, and original
   success criteria before starting. Do not lower criteria during the loop.
2. Validate without initializing state:

   ```bash
   uv run python "$AUTORESEARCH_SKILL_DIR/scripts/control.py" validate --config autoresearch.json
   ```

3. Inspect the native runtime's delegation capabilities and supported model /
   effort settings. Validate **all** configured round variants before dispatch.
   Omitted settings inherit native defaults; pass explicit settings to the
   native task API. If unsupported or unverifiable, stop and explain; never
   silently substitute settings. If the host requires a fresh context to
   override models, send a self-contained task with the necessary constraints.
   Record actual settings only when reported; requested settings are not proof
   of actual execution settings. A discovered mismatch stops further dispatch.
4. Inspect the project and its validation commands. Code-changing loops require
   a clean primary Git worktree; ignore only loop-owned `.aimx` state. Preserve
   existing user edits. Use native worktree support or Git for isolation.
5. Read the research state and relevant evidence. Initialize a loop and save
   its returned `loop_dir`. If the goal is already met, checkpoint a
   criterion-by-criterion assessment without launching a round:

   ```bash
   uv run python "$AUTORESEARCH_SKILL_DIR/scripts/control.py" init --config autoresearch.json
   ```

## Execute each round

Read [references/research-handoff.md](references/research-handoff.md) before
writing Research State. Only the lead performs the following orchestration.

1. Refresh context, Agenda, and Frontier for this goal. Respect human pauses
   and rejections. `research next` selects existing work; the lead creates any
   new tasks. Bootstrap empty state with a user-goal Frontier direction, not a
   fabricated empirical Finding. Create/activate the relevant Agenda tasks.
2. Form one bounded, independent task for each configured agent. Give each
   its own worktree from the current baseline, absolute data/Aim paths,
   separate output locations, relevant Findings, constraints, and result file.
   Persist task files and the baseline before dispatch.
   Read-only tasks without Git may use separate task/output directories and a
   stable evidence snapshot ID as `baseline_ref`; never modify shared inputs.
3. Reserve the round **before the first spawn attempt**:

   ```bash
   uv run python "$AUTORESEARCH_SKILL_DIR/scripts/control.py" begin-round \
     --loop-dir "$LOOP_DIR" --round 1 --baseline-ref "$BASELINE_COMMIT"
   ```

   A started round consumes budget even if spawning/training fails. Each slot
   launches at most once. Do not add unconfigured agents, recursively delegate,
   or retry a failed slot within that round. Queue pending slots in waves when
   host capacity is smaller than the configured list; this remains one round.
4. Before each native spawn, checkpoint its task/worktree and `reserved`
   dispatch state. After spawn, checkpoint `started` and the native task ID.
   Collect structured results into `finished` slots. One failure does not
   cancel the other tasks. Preserve failures as evidence for later rounds.
5. Review results against the goal and refresh human steering before integrating.
   For code changes, cherry-pick selected
   candidate commits into a separate integration worktree based on this round's
   baseline, then run project validation there. Exclude unresolved conflicts
   and failed candidates. Verify the primary HEAD is still the recorded
   baseline and the primary worktree is still clean before a local fast-forward
   merge. If user edits appeared, preserve candidates, mark the loop blocked,
   and stop integration. Do not push or discard user changes. Record the merged
   commit; it becomes the next round's baseline.
6. Refresh Research State. Author **one aggregate ResearchUpdate as the lead**,
   with real evidence, proposed Findings, Lineage, and Agenda outcomes. Persist
   its exact file and stable `client_update_id`, dry-run, then commit. Record
   the returned commit ID and revision. A failed round still gets an honest
   failure annotation / Agenda outcome, not invented run evidence.
7. Evaluate every original success criterion using inspected evidence and
   rationale. A successful task or improved metric alone is not goal completion.
   Persist the fresh assessment and mark the round complete only after all
   outcomes, integration, and the aggregate ResearchUpdate are reconciled.
   The helper derives `goal_met` or `budget_exhausted`; inspect `status` before
   starting another round. Goal success takes precedence on the last round.

## Explicit resume and finish

Resume only the loop requested by the user. Use `status --loop-dir ... --config
autoresearch.json` to verify the original config and inspect pending work.
Reuse the recorded budget, tasks, worktrees, result files, merged commits, and
update receipts. Never initialize a replacement loop to reset its budget.

- `begin-round` is idempotent registration, **not** permission to respawn.
  Reconnect to known native tasks; a reserved/started slot must not be launched
  again. Unknown process state means `blocked` or `interrupted` with a reason.
- Reconcile a completed merge against its saved Git refs before applying any
  changes again. Reconcile an uncertain ResearchUpdate using its exact file and
  idempotency key. See the handoff reference for revision conflicts.
- After reconciliation, explicitly checkpoint `status: "running"` to resume
  a blocked/interrupted loop. Lost tasks can be recorded as interrupted;
  a fresh attempt belongs to a later budgeted round.
- Never delete a writer lock until its recorded process is confirmed stopped.
  Preserve candidate commits, diffs, configs, results, and logs before cleaning
  worktrees. Keep commits reachable with local `codex/autoresearch-...` branches
  or Git bundles, and save artifact files outside disposable checkouts.
  Do not remove worktrees with unknown live experiments.

Finish with goal status, started/max rounds, criterion evidence or remaining
gaps, merged changes, useful Findings, failures, and the checkpoint location.
Round budget is not a token/cost/time or GPU-resource budget. Checkpoints do
not provide a scheduler or automatic process recovery.
