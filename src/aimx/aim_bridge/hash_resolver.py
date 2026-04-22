from __future__ import annotations

import re
from pathlib import Path

try:
    from aim import Repo
except ModuleNotFoundError:  # aim not installed; errors surface at call time
    Repo = None  # type: ignore[assignment,misc]

# Matches: run.hash == 'x'  run.hash=='x'  run.hash == "x"  run.hash != 'x'
# Capture groups: (1) operator token  (2) quote char  (3) hash literal
_HASH_LITERAL_RE = re.compile(
    r"""(run\.hash\s*(?:==|!=)\s*)(['"])([0-9a-fA-F]+)\2"""
)

_FULL_HASH_LEN = 32


def resolve_hash_prefixes(expression: str, repo_path: Path) -> str:
    """Rewrite short run.hash literals in *expression* to full hashes.

    Rules:
    - A literal shorter than ``_FULL_HASH_LEN`` hex chars is treated as a prefix.
    - A literal of exactly ``_FULL_HASH_LEN`` chars passes through unchanged.
    - Matching is case-insensitive (input normalised to lower-case).
    - Ambiguous prefix  → ``ValueError`` listing candidate previews.
    - No matching run   → ``ValueError``.
    - No ``run.hash`` literal in *expression* → expression returned as-is
      without querying the repository.
    """
    if not _HASH_LITERAL_RE.search(expression):
        return expression

    if Repo is None:
        raise RuntimeError(
            "`aimx` requires the Python `aim` package in the current environment."
        )

    repo = Repo(str(repo_path))
    all_hashes: list[str] = repo.list_all_runs()

    def _replace(m: re.Match[str]) -> str:
        operator_token = m.group(1)
        quote = m.group(2)
        value = m.group(3).lower()

        if len(value) >= _FULL_HASH_LEN:
            return m.group(0)

        candidates = [h for h in all_hashes if h.startswith(value)]

        if not candidates:
            raise ValueError(
                f"Short hash '{m.group(3)}' did not match any run in the repository."
            )
        if len(candidates) > 1:
            preview = ", ".join(c[:12] for c in candidates[:5])
            suffix = f" (+{len(candidates) - 5} more)" if len(candidates) > 5 else ""
            raise ValueError(
                f"Short hash '{m.group(3)}' is ambiguous — matches {len(candidates)} runs: "
                f"{preview}{suffix}. Provide more characters."
            )

        return f"{operator_token}{quote}{candidates[0]}{quote}"

    return _HASH_LITERAL_RE.sub(_replace, expression)
