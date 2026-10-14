from __future__ import annotations

import json


_RESEARCH_USAGE = {
    "research": "aimx research [-h|--help] <state|update|context|agenda|next|overview|annotations|receipt|evidence> ...",
    "research state": "aimx research state [--repo PATH] [--json]",
    "research update": "aimx research update [--repo PATH] (--stdin|--file FILE) [--dry-run] [--json]",
    "research context": "aimx research context --objective TEXT --budget BYTES [--repo PATH] [--loop LOOP_ID] [--json]",
    "research agenda": "aimx research agenda [--repo PATH] [--status STATUS] [--json]",
    "research next": "aimx research next [--repo PATH] [--json]",
    "research overview": "aimx research overview [--repo PATH] [--loop LOOP_ID] [--json]",
    "research annotations": "aimx research annotations [--repo PATH] [--topic TEXT] [--loop LOOP_ID] [--json]",
    "research receipt": "aimx research receipt --client-update-id KEY [--fingerprint SHA256] [--repo PATH] [--json]",
    "research evidence": "aimx research evidence check --finding FINDING_ID [--repo PATH] [--json]",
    "research evidence check": "aimx research evidence check --finding FINDING_ID [--repo PATH] [--json]",
    "finding": "aimx finding [-h|--help] <ls|show|comment|accept|reject|assess> ...",
    "finding ls": "aimx finding ls [--repo PATH] [--topic TEXT] [--loop LOOP_ID] [--json]",
    "finding show": "aimx finding show FINDING_ID [--repo PATH] [--json]",
    "finding comment": "aimx finding comment FINDING_ID TEXT... [--repo PATH]",
    "finding accept": "aimx finding accept FINDING_ID [--reason TEXT] [--repo PATH]",
    "finding reject": "aimx finding reject FINDING_ID [--reason TEXT] [--repo PATH]",
    "finding assess": "aimx finding assess FINDING_ID [--status STATUS] [--confidence LEVEL] [--reason TEXT] [--repo PATH]",
    "lineage": "aimx lineage [-h|--help] <show|link|retract> ...",
    "lineage show": "aimx lineage show FINDING_ID [--repo PATH] [--json]",
    "lineage link": "aimx lineage link SOURCE RELATION_TYPE TARGET [--reason TEXT] [--repo PATH]",
    "lineage retract": "aimx lineage retract RELATION_ID --reason TEXT [--repo PATH]",
    "frontier": "aimx frontier [-h|--help] <show|lane-add|add|move|pause|retire> ...",
    "frontier show": "aimx frontier show [--repo PATH] [--json]",
    "frontier lane-add": "aimx frontier lane-add NAME [--description TEXT] [--repo PATH]",
    "frontier add": "aimx frontier add --lane LANE (--finding ID|--direction TEXT) --rationale TEXT [--priority N] [--repo PATH]",
    "frontier move": "aimx frontier move ITEM_ID --lane LANE [--repo PATH]",
    "frontier pause": "aimx frontier pause ITEM_ID [--reason TEXT] [--repo PATH]",
    "frontier retire": "aimx frontier retire ITEM_ID [--reason TEXT] [--repo PATH]",
}


_READ_ONLY = {"research", "research state", "research context", "research agenda", "research next", "research overview", "research annotations", "research receipt", "research evidence", "research evidence check", "finding", "finding ls", "finding show", "lineage", "lineage show", "frontier", "frontier show"}

_DESCRIPTIONS = {command: ("Read-only research command." if command in _READ_ONLY else "Explicitly writes Aimx research state; native Aim data is unchanged.") for command in _RESEARCH_USAGE}
_OPTIONS = {command: ["--repo PATH", "--json", "-h", "--help"] for command in _RESEARCH_USAGE}
_OPTIONS.update({"research": ["-h", "--help"], "finding": ["-h", "--help"], "lineage": ["-h", "--help"], "frontier": ["-h", "--help"], "research update": ["--repo PATH", "--stdin | --file FILE", "--dry-run", "--json", "-h", "--help"], "research context": ["--repo PATH", "--objective TEXT", "--budget BYTES", "--loop LOOP_ID", "--json", "-h", "--help"], "research agenda": ["--repo PATH", "--status STATUS", "--json", "-h", "--help"], "research overview": ["--repo PATH", "--loop LOOP_ID", "--json", "-h", "--help"], "research annotations": ["--repo PATH", "--topic TEXT", "--loop LOOP_ID", "--json", "-h", "--help"], "research receipt": ["--repo PATH", "--client-update-id KEY", "--fingerprint SHA256", "--json", "-h", "--help"], "research evidence": ["check", "--repo PATH", "--finding FINDING_ID", "--json", "-h", "--help"], "research evidence check": ["--repo PATH", "--finding FINDING_ID", "--json", "-h", "--help"], "finding ls": ["--repo PATH", "--topic TEXT", "--loop LOOP_ID", "--json", "-h", "--help"], "finding show": ["FINDING_ID", "--repo PATH", "--json", "-h", "--help"], "finding comment": ["FINDING_ID", "TEXT...", "--repo PATH"], "finding accept": ["FINDING_ID", "--reason TEXT", "--repo PATH"], "finding reject": ["FINDING_ID", "--reason TEXT", "--repo PATH"], "finding assess": ["FINDING_ID", "--status STATUS", "--confidence LEVEL", "--reason TEXT", "--repo PATH"], "lineage show": ["FINDING_ID", "--repo PATH", "--json", "-h", "--help"], "lineage link": ["SOURCE", "RELATION_TYPE", "TARGET", "--reason TEXT", "--repo PATH"], "lineage retract": ["RELATION_ID", "--reason TEXT", "--repo PATH"], "frontier show": ["--repo PATH", "--json", "-h", "--help"], "frontier lane-add": ["NAME", "--description TEXT", "--repo PATH"], "frontier add": ["--lane LANE", "--finding ID | --direction TEXT", "--rationale TEXT", "--priority N", "--repo PATH"], "frontier move": ["ITEM_ID", "--lane LANE", "--repo PATH"], "frontier pause": ["ITEM_ID", "--reason TEXT", "--repo PATH"], "frontier retire": ["ITEM_ID", "--reason TEXT", "--repo PATH"]})
_EXAMPLES = {command: usage for command, usage in _RESEARCH_USAGE.items()}
_EXAMPLES.update({"research overview": "aimx research overview --repo ./study --loop study-1 --json", "research annotations": "aimx research annotations --repo ./study --topic accuracy --json", "research receipt": "aimx research receipt --client-update-id round-1 --repo ./study --json", "research evidence": "aimx research evidence check --finding f_123 --repo ./study --json", "research evidence check": "aimx research evidence check --finding f_123 --repo ./study --json", "finding ls": "aimx finding ls --repo ./study --topic accuracy --json", "finding comment": "aimx finding comment f_123 Verify the extended sample window --repo ./study", "finding accept": "aimx finding accept f_123 --reason Reproduced --repo ./study", "finding reject": "aimx finding reject f_123 --reason Confounded --repo ./study", "finding assess": "aimx finding assess f_123 --status candidate --confidence medium --repo ./study", "lineage link": "aimx lineage link f_new refines f_old --reason Later scope narrows the result --repo ./study", "lineage retract": "aimx lineage retract rel_123 --reason Superseded by explicit audit --repo ./study", "frontier lane-add": "aimx frontier lane-add robustness --description Follow-up directions --repo ./study", "frontier add": "aimx frontier add --lane robustness --direction Extend historical window --rationale Assess generalization --repo ./study", "frontier move": "aimx frontier move front_123 --lane robustness --repo ./study", "frontier pause": "aimx frontier pause front_123 --reason Awaiting review --repo ./study", "frontier retire": "aimx frontier retire front_123 --reason No longer relevant --repo ./study"})


def render_command_help(command: str, *, output_json: bool = False) -> str | None:
    usage = _RESEARCH_USAGE.get(command)
    if usage is None:
        return None
    description = _DESCRIPTIONS[command]
    mutation = command not in _READ_ONLY
    options = _OPTIONS[command]
    example = _EXAMPLES[command]
    value = {"schema_version": 1, "command": command, "usage": usage, "description": description, "mutation": mutation, "options": options, "examples": [example], "exit_codes": {"0": "success/help", "1": "internal error", "2": "input or state error", "3": "revision conflict"}}
    if output_json:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
    return "\n".join([usage, description, f"Mutation: {'yes' if mutation else 'no; read only'}", "Options: " + ", ".join(options), "Example: " + example, "Exit codes: 0 success/help; 1 internal; 2 input/state; 3 revision conflict", "Help does not read the repository."])


def render_help() -> str:
    return "\n".join(
        [
            "aimx - a safe companion CLI for native Aim",
            "",
            "Owned commands:",
            "  help       Show this help message",
            "  version    Show the aimx version and detected native Aim version",
            "  doctor     Show native Aim availability and passthrough readiness",
            "  query      Query metrics, images, or run params from a local Aim repository",
            "             Usage: aimx query <metrics|images|params> [<expression>] [--repo <path>]",
            "             Options: --json  --oneline  --no-color  --verbose",
            "                      --steps start:end | --epochs start:end  (mutually exclusive)",
            "                      --head N  --tail N  --every K",
            "                      --max-images N  (images only, TTY rendering cap; default 6)",
            "                      --param KEY  (params only, repeatable selected parameter)",
            "             Repo defaults to the current directory; paths may point at either",
            "             the repo root or its .aim directory",
            "             If expression is omitted, aimx uses: run.hash != ''",
            "             Short run hashes in the expression are transparently expanded.",
            "             Params output includes run duration when available.",
            "             Example: aimx query metrics --repo data",
            "             Example: aimx query images --repo data --plain",
            "             Example: aimx query params --repo data --json",
            "             Example: aimx query metrics \"run.hash=='eca37394'\" --repo data",
            "             Example: aimx query images \"images\" --repo data --epochs 10:50",
            "             Example: aimx query images \"images\" --repo data --head 10",
            "             Example: aimx query params \"run.experiment=='cloud-segmentation'\" --repo data --param hparam.lr",
            "  trace      Plot metric time-series or distribution histograms from a local Aim repository",
            "             Usage: aimx trace [<expression>] [--repo <path>]",
            "             Usage: aimx trace distribution [<expression>] [--repo <path>]",
            "             Options: --table  --csv  --json",
            "                      --steps start:end  (e.g. --steps 100:500, :50, 100:)",
            "                      --step N  (distribution visual mode only; nearest tracked step)",
            "                      --head N  --tail N  --every K",
            "                      --width W  --height H  --no-color",
            "             Repo defaults to the current directory.",
            "             If expression is omitted, aimx uses: run.hash != ''",
            "             Short run hashes in the expression are transparently expanded.",
            "             Example: aimx trace --repo data --head 5",
            "             Example: aimx trace distribution --repo data --head 2 --no-color",
            "             Example: aimx trace \"metric.name=='loss'\" --repo data --steps 100:500",
            "             Example: aimx trace distribution \"distribution.name != ''\" --repo data --step 12300",
            "             Example: aimx trace distribution \"distribution.name=='weights'\" --repo data --table",
            "             Example: aimx trace distribution \"distribution.name=='weights'\" --repo data --json",
            "  research   Read or update durable Aimx research state and agent context",
            "             Usage: aimx research <state|update|context|agenda|next|overview|annotations|receipt|evidence> ...",
            "             Reads are side-effect free; explicit update writes only .aimx/research.",
            "             Example: aimx research state --repo data --json",
            "             Example: aimx research context --repo data --objective \"improve accuracy\" --budget 12000 --json",
            "             Example: aimx research update --repo data --file update.json --json",
            "  finding    Inspect or steer durable research findings",
            "             Usage: aimx finding <ls|show|comment|accept|reject|assess> ...",
            "             Reads do not write .aimx; comment/accept/reject/assess are explicit writes.",
            "             Example: aimx finding ls --repo data --json",
            "             Example: aimx finding accept <finding-id> --reason \"carry forward\" --repo data",
            "  lineage    Inspect or edit relationships between findings",
            "             Usage: aimx lineage <show|link|retract> ...",
            "             Example: aimx lineage show <finding-id> --repo data --json",
            "             Example: aimx lineage link <source-id> challenges <target-id> --repo data",
            "  frontier   Inspect or steer project-defined research frontier",
            "             Usage: aimx frontier <show|lane-add|add|move|pause|retire> ...",
            "             Example: aimx frontier show --repo data --json",
            "             Example: aimx frontier pause <item-id> --reason \"review\" --repo data",
            "",
            "All other commands are delegated to native `aim`.",
        ]
    )
