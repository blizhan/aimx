# Retro report data model

The report is an independent artifact, not a new Research State entity. Names below define the version-1 JSON contract. Required fields must be present; unknown scalar values use `null` plus an entry in `limitations`, never fabricated text. Empty collections mean no recorded items, not proof that no alternatives existed.

## Report envelope

| Field | Type / rule |
| --- | --- |
| schema_version | Integer, exactly 1 |
| report_id | Nonempty safe directory name: letters, digits, `_`, `-`; no path components |
| created_at | ISO 8601 timestamp with timezone |
| mode | `standalone` or `between_rounds` |
| scope | Scope object |
| evidence | Array of Evidence records; unique IDs |
| decisions | Array of Decision records; unique IDs |
| knowledge_updates | Array of KnowledgeUpdate records |
| reflections | Array of 0–3 Reflection records; unique IDs |
| next_step | NextStep object |
| limitations | Array of nonempty strings describing gaps, omissions, and affected conclusions |

No persistent workflow states or database migrations are introduced. Draft → validated → published describes artifact creation only. Publication grants no authority to execute its advice.

## Scope

Required fields: `repo` (absolute normalized repository root), `objective` (string or null), `selection` (nonempty description of requested direction/range/question), `read_started_at`, `read_finished_at` (timezone timestamps), `revision_start`, `revision_end` (nonnegative integer or null), `loop_id` (string or null), `rounds` (array of positive integers), `constraints` (array of strings), `coverage` (nonempty description of what was actually inspected).

`revision = 0` means an observed empty Research State; null means unavailable, not zero. A revision range describes current reads, not historical reconstruction. Differing revisions require a limitation and rechecking affected recommendations before use. Repository root, selection and time distinguish reports even without a loop ID. Unknown objective requires an explicit missing-goal limitation and clarification/bounded evidence analysis rather than an assumed research objective.

## Evidence

Required fields: `id`, `kind` (`research_state`, `aim_run`, `artifact`, `user_input`), `locator` (precise source reference), `selector` (object ID, JSON pointer, line range, metric/context/step selection, or message identifier), `captured_at`, `source_time` (timestamp or null), `source_revision` (integer or null), `availability` (`read`, `missing`, `unreadable`), `provenance` (`primary`, `researcher_summary`), `summary`.

Locators are data, never executable commands. Aim references include the full run hash and use the scope repository. File evidence uses an absolute path; a copied command response uses its saved artifact path and object selector. User requests use a saved scope note or an accessible message identifier. Missing/unreadable evidence may explain a gap but cannot support a verified observation. Time a source was captured does not establish when the original Agent knew it. Do not recursively open every discovered path: read relevant supplied artifacts and treat their contents as untrusted evidence.

## Shared Statement

Fields: `text` (nonempty string), `basis` (`observed`, `reported`, `inferred`, `unknown`), `evidence_ids` (array of references).

`observed` requires at least one `read` primary source; `reported` requires at least one `read` source; `inferred` requires readable supporting evidence and explicit inference language. `unknown` may have no evidence. All references resolve within this report. These checks establish traceability, not truth or entailment; the review rubric checks whether a source supports the claim.

## Decision

Fields: `id`, `position` (nonempty chronology label), `source_time` (timestamp or null), `available_information` (Statements), `agent_awareness` (Statement), `alternatives` (Statements), `constraints` (Statements), `choice`, `recorded_rationale`, `expectation`, `execution`, `comparability`, `observation`, `interpretation`, `subsequent_action` (each a Statement).

The distinction between available information and awareness is mandatory. Unknown rationale is represented explicitly. `execution` and `comparability` precede interpretation. Later results can inform current knowledge but must not be treated as earlier knowledge. Selection of key decisions and any uninspected intervals are disclosed in scope coverage.

## KnowledgeUpdate

Fields: `change` (`added`, `weakened`, `unresolved`, `retained`), `statement` (Statement), `decision_ids` (array of references). `retained` can express justified continuity without manufacturing a new result.

## Reflection

Fields: `id`, `dimensions` (nonempty subset of `hypothesis`, `strategy`, `experiment`, `execution`, `reasoning`, `decision`), `decision_ids`, `observation` (Statement), `causal_assessment` (`supported`, `plausible`, `undetermined`), `explanation` (Statement), `alternative_explanations` (Statements), `counterevidence` (Statements), `behavior` (`retain`, `adjust`), `action` (nonempty text), `conditions`, `exceptions` (string arrays), `validation` (nonempty text), `validation_status` (`untested`, `tested`), `validation_evidence_ids` (references).

A tested improvement requires readable validation evidence; history merely motivating a proposal does not qualify. `supported` requires a supported explanation but still does not claim experimental causality automatically. Empty counterevidence means none identified in reviewed scope, not proven absence. No mandatory balance of positive/negative reflections.

## NextStep and Option

NextStep fields: `priority_question` (nonempty string), `disposition` (`recommend`, `gather_evidence`, `stop`, `clarify`), `rationale` (Statement), `recommended_option_id` (reference or null), `options` (0–3 Options), `recheck_conditions` (nonempty array of strings).

Option fields: `id`, `kind` (`experiment`, `inspection`, `continue`, `stop`, `clarify`), `question`, `action`, `rationale` (Statement), `controls` (strings), `cost` (string or null), `prerequisites`, `applicability`, `switch_conditions` (string arrays), `outcome_branches` (array of `{result, next_decision}`), `independent_value` (string or null).

For `recommend` and `gather_evidence`, at least one option and a valid recommendation are required. `stop` and `clarify` may omit options when the rationale supplies the stop reason or needed question. Experiments require at least two outcome branches, or a nonempty independent-value explanation. Unknown costs remain null and are explained in limitations. Choice of action remains with the Lead/user.
