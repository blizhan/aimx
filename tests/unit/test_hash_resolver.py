from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aimx.aim_bridge.hash_resolver import resolve_hash_prefixes, _FULL_HASH_LEN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FULL_A = "a" * _FULL_HASH_LEN
_FULL_B = "b" * _FULL_HASH_LEN
_FULL_C = "abc123" + "0" * (_FULL_HASH_LEN - 6)


def _repo(hashes: list[str]):
    """Return a mock Aim Repo whose list_all_runs() returns *hashes*."""
    mock_repo = MagicMock()
    mock_repo.list_all_runs.return_value = hashes
    return mock_repo


def _patch_repo(hashes: list[str]):
    return patch(
        "aimx.aim_bridge.hash_resolver.Repo",
        return_value=_repo(hashes),
    )


# ---------------------------------------------------------------------------
# Expression contains no run.hash → no repo call needed
# ---------------------------------------------------------------------------


def test_expression_without_run_hash_is_returned_unchanged(tmp_path) -> None:
    expr = "metric.name == 'loss'"
    assert resolve_hash_prefixes(expr, tmp_path) == expr


def test_expression_without_run_hash_does_not_call_repo(tmp_path) -> None:
    with patch("aimx.aim_bridge.hash_resolver.Repo") as mock_cls:
        resolve_hash_prefixes("metric.name == 'loss'", tmp_path)
        mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Full-length hash passes through unchanged
# ---------------------------------------------------------------------------


def test_full_length_hash_is_not_rewritten(tmp_path) -> None:
    expr = f"run.hash == '{_FULL_A}'"
    with _patch_repo([_FULL_A, _FULL_B]):
        result = resolve_hash_prefixes(expr, tmp_path)
    assert result == expr


# ---------------------------------------------------------------------------
# Unique prefix → expanded to full hash
# ---------------------------------------------------------------------------


def test_unique_prefix_is_expanded_to_full_hash(tmp_path) -> None:
    prefix = "abc123"
    expr = f"run.hash == '{prefix}'"
    with _patch_repo([_FULL_C, _FULL_B]):
        result = resolve_hash_prefixes(expr, tmp_path)
    assert _FULL_C in result
    # The short literal should now be replaced by the full hash
    assert f"'{prefix}'" not in result


def test_prefix_with_spaces_around_operator_is_expanded(tmp_path) -> None:
    prefix = "abc123"
    expr = f"run.hash == '{prefix}'"
    with _patch_repo([_FULL_C]):
        result = resolve_hash_prefixes(expr, tmp_path)
    assert _FULL_C in result


def test_prefix_without_spaces_around_operator_is_expanded(tmp_path) -> None:
    prefix = "abc123"
    expr = f"run.hash=='{prefix}'"
    with _patch_repo([_FULL_C]):
        result = resolve_hash_prefixes(expr, tmp_path)
    assert _FULL_C in result


# ---------------------------------------------------------------------------
# Quote style
# ---------------------------------------------------------------------------


def test_double_quoted_prefix_is_expanded(tmp_path) -> None:
    prefix = "abc123"
    expr = f'run.hash == "{prefix}"'
    with _patch_repo([_FULL_C]):
        result = resolve_hash_prefixes(expr, tmp_path)
    assert _FULL_C in result
    assert '"' in result


def test_single_quoted_prefix_is_expanded(tmp_path) -> None:
    prefix = "abc123"
    expr = f"run.hash == '{prefix}'"
    with _patch_repo([_FULL_C]):
        result = resolve_hash_prefixes(expr, tmp_path)
    assert _FULL_C in result
    assert "'" in result


# ---------------------------------------------------------------------------
# != operator
# ---------------------------------------------------------------------------


def test_not_equal_operator_is_supported(tmp_path) -> None:
    prefix = "abc123"
    expr = f"run.hash != '{prefix}'"
    with _patch_repo([_FULL_C]):
        result = resolve_hash_prefixes(expr, tmp_path)
    assert _FULL_C in result
    assert "!=" in result


# ---------------------------------------------------------------------------
# Case-insensitive prefix matching
# ---------------------------------------------------------------------------


def test_uppercase_prefix_matches_lowercase_hash(tmp_path) -> None:
    full = "abc123" + "0" * (_FULL_HASH_LEN - 6)
    expr = f"run.hash == 'ABC123'"
    with _patch_repo([full]):
        result = resolve_hash_prefixes(expr, tmp_path)
    assert full in result


# ---------------------------------------------------------------------------
# Error: no match
# ---------------------------------------------------------------------------


def test_unmatched_prefix_raises_value_error(tmp_path) -> None:
    expr = "run.hash == 'deadbeef'"
    with _patch_repo([_FULL_A, _FULL_B]):
        with pytest.raises(ValueError, match="did not match"):
            resolve_hash_prefixes(expr, tmp_path)


def test_error_message_contains_original_short_hash(tmp_path) -> None:
    expr = "run.hash == 'deadbeef'"
    with _patch_repo([_FULL_A]):
        with pytest.raises(ValueError, match="deadbeef"):
            resolve_hash_prefixes(expr, tmp_path)


# ---------------------------------------------------------------------------
# Error: ambiguous
# ---------------------------------------------------------------------------


def test_ambiguous_prefix_raises_value_error(tmp_path) -> None:
    full1 = "abc" + "1" * (_FULL_HASH_LEN - 3)
    full2 = "abc" + "2" * (_FULL_HASH_LEN - 3)
    expr = "run.hash == 'abc'"
    with _patch_repo([full1, full2]):
        with pytest.raises(ValueError, match="ambiguous"):
            resolve_hash_prefixes(expr, tmp_path)


def test_ambiguous_error_lists_candidates(tmp_path) -> None:
    full1 = "abc" + "1" * (_FULL_HASH_LEN - 3)
    full2 = "abc" + "2" * (_FULL_HASH_LEN - 3)
    expr = "run.hash == 'abc'"
    with _patch_repo([full1, full2]):
        with pytest.raises(ValueError, match="abc"):
            resolve_hash_prefixes(expr, tmp_path)


# ---------------------------------------------------------------------------
# Multiple run.hash literals in one expression
# ---------------------------------------------------------------------------


def test_multiple_hash_literals_are_all_resolved(tmp_path) -> None:
    full1 = "aaaa" + "0" * (_FULL_HASH_LEN - 4)
    full2 = "bbbb" + "0" * (_FULL_HASH_LEN - 4)
    expr = "run.hash == 'aaaa' or run.hash == 'bbbb'"
    with _patch_repo([full1, full2]):
        result = resolve_hash_prefixes(expr, tmp_path)
    assert full1 in result
    assert full2 in result
    assert "'aaaa'" not in result
    assert "'bbbb'" not in result
