from __future__ import annotations

import sys

from aimx.cli import run_cli


def main(argv: list[str] | None = None) -> int:
    return run_cli(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
