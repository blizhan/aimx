#!/usr/bin/env python3
"""Collect a read-only aimx experiment snapshot for autoresearch logs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class CommandPayload:
    argv: list[str]
    ok: bool
    data: Any | None = None
    message: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"argv": self.argv, "ok": self.ok}
        if self.data is not None:
            payload["data"] = self.data
        if self.message:
            payload["message"] = self.message
        if self.error:
            payload["error"] = self.error
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect params, metric summaries, optional traces, and image metadata with aimx."
    )
    parser.add_argument("--repo", default=".", help="Aim repo root or .aim metadata path.")
    parser.add_argument(
        "--base-expr",
        default="run.hash != ''",
        help="AimQL run scope used for params and generated metric expressions.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Metric name to summarize; repeatable. Defaults to all metrics when no metric expression is supplied.",
    )
    parser.add_argument(
        "--metric-expr",
        action="append",
        default=[],
        help="Full AimQL metric expression to summarize; repeatable.",
    )
    parser.add_argument(
        "--trace-metric",
        action="append",
        default=[],
        help="Metric name to collect trace samples for; repeatable.",
    )
    parser.add_argument(
        "--trace-expr",
        action="append",
        default=[],
        help="Full AimQL trace expression; repeatable.",
    )
    parser.add_argument(
        "--trace-tail",
        type=int,
        default=50,
        help="Tail sample count for trace commands. Use 0 to disable tail sampling.",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Parameter key to select; repeatable. Omit to let aimx discover params.",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="Also collect image metadata with aimx query images.",
    )
    parser.add_argument(
        "--image-expr",
        default="images",
        help="AimQL image expression used when --include-images is set.",
    )
    parser.add_argument(
        "--image-head",
        type=int,
        default=20,
        help="Limit image metadata rows when --include-images is set.",
    )
    parser.add_argument(
        "--aimx",
        default=f"{sys.executable} -m aimx",
        help='Launcher for aimx, for example "aimx", "uv run aimx", or "python -m aimx".',
    )
    parser.add_argument("--timeout", type=int, default=60, help="Per-command timeout in seconds.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def aimql_string(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def metric_expression(base_expr: str, metric_name: str) -> str:
    return f"({base_expr}) and metric.name == {aimql_string(metric_name)}"


def run_json(base_cmd: list[str], args: list[str], timeout: int) -> CommandPayload:
    argv = [*base_cmd, *args]
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return CommandPayload(argv=argv, ok=False, error=str(exc))
    except subprocess.TimeoutExpired as exc:
        return CommandPayload(argv=argv, ok=False, error=f"Timed out after {timeout}s: {exc}")

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        return CommandPayload(
            argv=argv,
            ok=False,
            message=stdout or None,
            error=stderr or f"Command exited with status {completed.returncode}",
        )

    if not stdout:
        return CommandPayload(argv=argv, ok=True, data=None)

    try:
        return CommandPayload(argv=argv, ok=True, data=json.loads(stdout))
    except json.JSONDecodeError:
        return CommandPayload(argv=argv, ok=True, message=stdout)


def collect_snapshot(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    base_cmd = shlex.split(args.aimx)
    metric_exprs = list(args.metric_expr)
    metric_exprs.extend(metric_expression(args.base_expr, name) for name in args.metric)
    if not metric_exprs:
        metric_exprs.append(f"({args.base_expr}) and metric.name != ''")

    trace_exprs = list(args.trace_expr)
    trace_exprs.extend(metric_expression(args.base_expr, name) for name in args.trace_metric)

    params_args = ["query", "params", args.base_expr, "--repo", args.repo, "--json"]
    for key in args.param:
        params_args.extend(["--param", key])

    params = run_json(base_cmd, params_args, args.timeout)

    metrics = [
        run_json(base_cmd, ["query", "metrics", expr, "--repo", args.repo, "--json"], args.timeout)
        for expr in metric_exprs
    ]

    trace_common = ["--repo", args.repo, "--json"]
    if args.trace_tail > 0:
        trace_common.extend(["--tail", str(args.trace_tail)])
    traces = [
        run_json(base_cmd, ["trace", expr, *trace_common], args.timeout)
        for expr in trace_exprs
    ]

    images = None
    if args.include_images:
        image_args = [
            "query",
            "images",
            args.image_expr,
            "--repo",
            args.repo,
            "--json",
            "--head",
            str(args.image_head),
        ]
        images = run_json(base_cmd, image_args, args.timeout)

    snapshot: dict[str, Any] = {
        "repo": args.repo,
        "base_expr": args.base_expr,
        "read_only": True,
        "params": params.as_dict(),
        "metrics": [item.as_dict() for item in metrics],
        "traces": [item.as_dict() for item in traces],
        "images": images.as_dict() if images else None,
    }

    failures = [params, *metrics, *traces]
    if images:
        failures.append(images)
    ok = all(item.ok for item in failures)
    return snapshot, ok


def main() -> int:
    args = parse_args()
    snapshot, ok = collect_snapshot(args)
    indent = 2 if args.pretty else None
    print(json.dumps(snapshot, indent=indent, sort_keys=args.pretty))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
