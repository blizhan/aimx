# Loop configuration, dispatch, and checkpoints

The helper is a standalone Python 3.10–3.12 script using only the standard
library. It never imports Aim/Aimx, launches processes, creates worktrees,
integrates Git changes, or writes the research database. All output is JSON;
successful commands return exit 0, input/state/I/O errors return exit 2 with
`ok: false` and `error.code/message`. CLI argument errors use argparse's exit 2.
Only the lead runs its mutating commands.

## Configuration

Copy [../assets/config.example.json](../assets/config.example.json) into the
target project as `autoresearch.json`, then edit its paths, goal, budget, and
model IDs. Paths resolve relative to the config file, expand `~`, and normalize
to absolute paths. Both directories must already exist. `aim_repo` may be the
repo root or its `.aim` directory; the latter normalizes to the parent root.
An empty Research State is valid and does not require invented Aim evidence.

| Field | Contract |
| --- | --- |
| `schema_version` | Required integer `1`. |
| `project_root`, `aim_repo` | Required existing directory paths. |
| `goal.description` | Required nonempty target description. |
| `goal.success_criteria` | Required nonempty array of distinct text criteria; immutable after initialization. |
| `max_rounds` | Required positive integer; a failed started round still counts. |
| `default_agents` | Required nonempty list of agent settings. |
| `round_overrides` | Optional object with canonical 1-based round numbers as keys and `{ "agents": [...] }` values. Each list replaces the entire default list for that round. |
| `context_budget_bytes` | Optional positive integer, default 12000; passed to `research context --budget`. |

Each agent has a unique `id` within its round and optional nonempty `model` and
`reasoning_effort` strings. IDs use letters, digits, `_` or `-`, start with a
letter/digit, and are at most 64 characters. Settings omitted from an override
agent inherit **host defaults**, not fields from a same-named default agent.

Model/effort strings are native runtime values, not a maintained global enum.
The sample's explicit Codex model is illustrative; replace it for other hosts.
Validation checks shape, not runtime availability. The lead must separately
verify native support and propagate each explicit setting unchanged, or use
a documented equivalent field mapping. No silent downgrade is permitted.
Unsupported fields and duplicate JSON keys are rejected to catch typos in
budgets or settings.

## Dispatch template — include in every task

Fill this with the round's actual values and persist it before spawning:

```text
You are subagent {agent_id} in loop {loop_id}, round {round}.
Goal and original success criteria: {goal}
Task / hypothesis / bounded experiment plan: {task}
Relevant Findings, controls, and constraints: {context}
Your worktree and baseline commit: {worktree}, {baseline_ref}
Project environment and validation commands: {commands}
Absolute data paths and Aim repository: {data_paths}, {aim_repo}
Your separate output directory and result file: {outputs}, {result_file}
Requested native model / reasoning effort: {requested_settings}

Aim run metadata for this dispatch (fill every placeholder before spawning):
  run.experiment = "ar-{loop_id}"
  run["aimx"] = {"loop_id": "{loop_id}", "round": {round},
                 "{branch_or_task_key}": "{branch_or_task_id}", "role": "{role}",
                 "hypothesis": "{hypothesis}"}
  tags = ["loop:{loop_id}", "round:{round}"]
Use `branch` or `task_id` as `{branch_or_task_key}`. Assign one concrete role
(`baseline`, `candidate`, or `control`) and one-line hypothesis per dispatch;
every run created by this dispatch carries these values.

Set this metadata when the training script/logger creates its Aim run, before
training starts. If a run already exists, the only sanctioned backfill is via
the Aim SDK `Run(run_hash=..., repo=...)`, writing only `run["aimx"]` and the
two convention tags. This backfill cannot repair `run.experiment`; if it was
not set at run creation, report the run as nonconforming. Never change metrics
or other params during backfill.

You are not alone in this project. Work only in your assigned worktree; preserve
other agents' and the user's changes. Do not integrate into the primary worktree.
You may read Aimx state and evidence and run the normal project training/logger.
The lead alone writes Research State and loop checkpoints. Do not generate or
submit a final ResearchUpdate; do not call research update or mutating finding,
frontier, or lineage commands; do not modify shared .aimx files or run control.py.
Do not spawn more agents or retry a failed assignment as a new task.
Return result JSON and preserve your evidence and candidate commit/diff before
any worktree cleanup. Report failure/uncertainty honestly. Actual model settings
must come from native runtime metadata; omit them when unknown.
```

## Agent result

```json
{
  "status": "succeeded",
  "summary": "The candidate improved validation accuracy under the fixed seed.",
  "evidence": ["aim_run:FULL_RUN_HASH", "/absolute/path/to/validation.json"],
  "aim_runs": [{
    "hash": "FULL_RUN_HASH",
    "experiment": "ar-accuracy-study",
    "aimx": {
      "loop_id": "accuracy-study",
      "round": 1,
      "task_id": "optimizer",
      "role": "candidate",
      "hypothesis": "The candidate improves validation accuracy under the fixed seed."
    },
    "tags": ["loop:accuracy-study", "round:1"]
  }],
  "candidate_ref": "CANDIDATE_COMMIT_SHA",
  "actual_settings": {"model": "gpt-6.1-sol", "reasoning_effort": "high"}
}
```

Required: `status` (`succeeded`, `failed`, `interrupted`), nonempty `summary`,
and `evidence` (array of nonempty reference strings, possibly empty for failure
or missing evidence). Include an `aim_runs` entry for every Aim run created and
an empty `aim_runs` array when no Aim runs were created. Each entry repeats the
exact run hash, experiment, `aimx` params, and convention tags for lead verification.
Optional: `candidate_ref` and `actual_settings`.
Use canonical `aim_run:<full-hash>` references for Aim evidence; use durable
absolute files or source URLs for other evidence. The lead verifies each run's
`run.experiment`, `aimx.*` params, and tags before relying on its findings. The
lead may apply the sanctioned metadata-only backfill above, then records the
verified canonical run references in the existing ResearchUpdate evidence
shape. Non-Aim evidence stays in artifacts/annotations; do not extend
ResearchUpdate with new evidence kinds.

## Helper commands

```bash
uv run python "$AUTORESEARCH_SKILL_DIR/scripts/control.py" validate --config autoresearch.json
uv run python "$AUTORESEARCH_SKILL_DIR/scripts/control.py" init --config autoresearch.json --loop-id accuracy-study
uv run python "$AUTORESEARCH_SKILL_DIR/scripts/control.py" begin-round \
  --loop-dir "$LOOP_DIR" --round 1 --baseline-ref "$BASELINE_COMMIT"
uv run python "$AUTORESEARCH_SKILL_DIR/scripts/control.py" checkpoint \
  --loop-dir "$LOOP_DIR" --file checkpoint.json
uv run python "$AUTORESEARCH_SKILL_DIR/scripts/control.py" status \
  --loop-dir "$LOOP_DIR" --config autoresearch.json
```

- `validate`: read-only normalized config, fingerprint, and effective agent
  lists for all rounds. No sidecar initialization.
- `init`: creates `<aim-repo>/.aimx/autoresearch/<loop-id>/state.json`.
  Omit `--loop-id` to generate one. Repeating the same ID/config returns existing
  state without resetting budget. A changed config is rejected.
- `begin-round`: registers the next round and baseline, consuming budget
  atomically before dispatch. Re-registering an existing round with the same
  baseline is read-only and does not authorize replay. Complete the prior
  round before starting another.
- `checkpoint`: applies a lead-authored JSON patch. Unknown agents and slot
  rewinds are rejected; original config, finished results, committed receipts,
  and completed rounds cannot be replaced. Artifact references accumulate.
- `status`: read-only state. `--config` on status/begin-round/checkpoint checks
  that a resume still uses the initialized config fingerprint.

The immutable config and consumed rounds remain in the checkpoint. Atomic file
replacement and an exclusive `.writer.lock` serialize writes; a crash may leave
a lock. Check its PID and any running control command before removing it.
The lock is not a scheduler lease or a guarantee about live training processes.

### Before spawning a slot

```json
{
  "round": 1,
  "agents": [{
    "id": "optimizer",
    "task_file": "/absolute/path/to/optimizer-task.txt",
    "worktree": "/absolute/path/to/optimizer-worktree",
    "dispatch_status": "reserved"
  }]
}
```

For read-only tasks without Git, `worktree` may identify the task's separate
working directory; use a stable evidence snapshot ID for the round baseline.
Code-changing tasks require a real isolated Git worktree.

Then checkpoint `dispatch_status: "started"` and the native `task_id`. A
reserved slot means a spawn may already have occurred. Never launch it again
after a crash. To finish, checkpoint `dispatch_status: "finished"` and `result`.
An undispatched slot can only finish as failed/interrupted, never succeeded.
Keep a stable task ID; use a later round for another attempt.

### Complete a round after lead aggregation

```json
{
  "round": 1,
  "agents": [{
    "id": "optimizer",
    "dispatch_status": "finished",
    "result": {
      "status": "succeeded",
      "summary": "Validated the selected optimizer change.",
      "evidence": ["/absolute/path/to/validation.json"]
    }
  }],
  "artifacts": ["/absolute/path/to/candidate.patch"],
  "merged_ref": "MERGED_COMMIT_SHA",
  "update_file": "/absolute/path/to/exact-research-update.json",
  "research_update": {
    "client_update_id": "accuracy-study-round-1-aggregate",
    "commit_id": "RETURNED_AIMX_COMMIT_ID",
    "revision": 2
  },
  "complete": true,
  "goal_assessment": [
    {
      "criterion": "Produce an evidence-backed comparison against the fixed baseline.",
      "met": true,
      "evidence": ["/absolute/path/to/comparison.json"],
      "rationale": "The comparison uses the same held-out split."
    },
    {
      "criterion": "Validate the selected changes and explain remaining uncertainty.",
      "met": false,
      "evidence": [],
      "rationale": "Replication under another seed is still missing."
    }
  ]
}
```

Other slots must also have finished records. `merged_ref` is optional when no
code was selected; the exact update file and ResearchUpdate receipt are required
for round completion, including failed rounds. Save draft update and integration
refs in separate checkpoints before external writes so an interrupted operation
can be reconciled. Record the integration candidate ref as an artifact before
merging; record `merged_ref` after verifying the actual primary HEAD.

Assessments match every original criterion, in its original order. A met
criterion requires evidence and a rationale. The helper validates this record;
it cannot independently prove the evidence or a research claim.

To block/interrupt, checkpoint `{ "status": "blocked", "stop_reason": "..." }`
or `interrupted`. After explicit resume and reconciliation, use
`{ "status": "running" }`. These transitions never refund budget or reset a
slot. With no active round, an evidence-backed assessment can mark an already
achieved goal without spending a round. Completed-round assessments derive
`goal_met` before `budget_exhausted`. Terminal loops cannot be reopened.
