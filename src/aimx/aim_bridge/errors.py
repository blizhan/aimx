from __future__ import annotations

_AIM_SQLALCHEMY_POOL_MARKERS = (
    "pool_size",
    "max_overflow",
    "NullPool",
    "create_engine()",
)


def is_aim_sqlalchemy_pool_error(message: str) -> bool:
    """Return True when Aim's SQLite engine setup is incompatible with SQLAlchemy."""
    lowered = message.lower()
    return all(marker.lower() in lowered for marker in _AIM_SQLALCHEMY_POOL_MARKERS[:2]) and (
        "nullpool" in lowered or "create_engine()" in lowered
    )


def format_query_evaluation_error(error: BaseException) -> str:
    message = str(error)
    if is_aim_sqlalchemy_pool_error(message):
        return (
            "Failed to evaluate query: incompatible SQLAlchemy version for Aim "
            f"({message}). Aim 3.28+ requires SQLAlchemy 2.0+ for SQLite metadata "
            "access. Upgrade with: uv add 'sqlalchemy>=2.0,<3'"
        )
    return f"Failed to evaluate query: {error}"


def format_trace_evaluation_error(error: BaseException) -> str:
    message = str(error)
    if is_aim_sqlalchemy_pool_error(message):
        return (
            "Failed to evaluate trace: incompatible SQLAlchemy version for Aim "
            f"({message}). Aim 3.28+ requires SQLAlchemy 2.0+ for SQLite metadata "
            "access. Upgrade with: uv add 'sqlalchemy>=2.0,<3'"
        )
    return f"Failed to evaluate trace: {error}"
