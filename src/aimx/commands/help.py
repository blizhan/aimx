from __future__ import annotations


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
            "             Usage: aimx query <metrics|images|params> <expression> [--repo <path>]",
            "             Options: --json  --oneline  --no-color  --verbose",
            "                      --steps start:end | --epochs start:end  (mutually exclusive)",
            "                      --head N  --tail N  --every K",
            "                      --max-images N  (images only, TTY rendering cap; default 6)",
            "                      --param KEY  (params only, repeatable selected parameter)",
            "             Repo defaults to the current directory; paths may point at either",
            "             the repo root or its .aim directory",
            "             Short run hashes in the expression are transparently expanded.",
            "             Example: aimx query metrics \"run.hash=='eca37394'\" --repo data",
            "             Example: aimx query images \"images\" --repo data --epochs 10:50",
            "             Example: aimx query images \"images\" --repo data --head 10",
            "             Example: aimx query params \"run.experiment=='cloud-segmentation'\" --repo data --param hparam.lr",
            "  trace      Plot a metric's time-series from a local Aim repository",
            "             Usage: aimx trace <expression> [--repo <path>]",
            "             Options: --table  --csv  --json",
            "                      --steps start:end  (e.g. --steps 100:500, :50, 100:)",
            "                      --head N  --tail N  --every K",
            "                      --width W  --height H  --no-color",
            "             Repo defaults to the current directory.",
            "             Short run hashes in the expression are transparently expanded.",
            "             Example: aimx trace \"metric.name=='loss'\" --repo data --steps 100:500",
            "",
            "All other commands are delegated to native `aim`.",
        ]
    )
