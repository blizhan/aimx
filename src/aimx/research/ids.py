from __future__ import annotations

import re
import uuid

PREFIXES = {
    "commit": "commit_",
    "event": "event_",
    "finding": "f_",
    "relation": "rel_",
    "annotation": "ann_",
    "lane": "lane_",
    "frontier": "front_",
    "agenda": "agenda_",
}

_LOCAL_REF_RE = re.compile(r"^\$[A-Za-z][A-Za-z0-9_-]*$")
_LOCAL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def new_id(kind: str) -> str:
    try:
        prefix = PREFIXES[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown research ID kind: {kind}") from exc
    return f"{prefix}{uuid.uuid4().hex}"


def is_local_ref(value: object) -> bool:
    return isinstance(value, str) and bool(_LOCAL_REF_RE.fullmatch(value))


def local_name(value: str) -> str:
    if not is_local_ref(value):
        raise ValueError(f"Not a valid local research reference: {value!r}")
    return value[1:]


def is_local_name(value: object) -> bool:
    return isinstance(value, str) and bool(_LOCAL_NAME_RE.fullmatch(value))


def expected_prefix(kind: str) -> str:
    try:
        return PREFIXES[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown research ID kind: {kind}") from exc
