from __future__ import annotations

from aimx.aim_bridge.errors import (
    format_query_evaluation_error,
    format_trace_evaluation_error,
    is_aim_sqlalchemy_pool_error,
)


def test_is_aim_sqlalchemy_pool_error_matches_linux_failure() -> None:
    message = (
        "Invalid argument(s) 'pool_size','max_overflow' sent to create_engine(), "
        "using configuration SQLiteDialect_pysqlite/NullPool/Engine."
    )
    assert is_aim_sqlalchemy_pool_error(message)


def test_format_query_evaluation_error_adds_upgrade_hint() -> None:
    error = TypeError(
        "Invalid argument(s) 'pool_size','max_overflow' sent to create_engine(), "
        "using configuration SQLiteDialect_pysqlite/NullPool/Engine."
    )
    formatted = format_query_evaluation_error(error)
    assert "Failed to evaluate query" in formatted
    assert "SQLAlchemy 2.0+" in formatted
    assert "uv add 'sqlalchemy>=2.0,<3'" in formatted


def test_format_query_evaluation_error_preserves_generic_message() -> None:
    error = ValueError("metric.name ==")
    assert format_query_evaluation_error(error) == "Failed to evaluate query: metric.name =="


def test_format_trace_evaluation_error_adds_upgrade_hint() -> None:
    error = TypeError(
        "Invalid argument(s) 'pool_size','max_overflow' sent to create_engine(), "
        "using configuration SQLiteDialect_pysqlite/NullPool/Engine."
    )
    formatted = format_trace_evaluation_error(error)
    assert "Failed to evaluate trace" in formatted
    assert "SQLAlchemy 2.0+" in formatted
