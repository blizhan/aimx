from __future__ import annotations

from aimx.router import route_args


def test_empty_invocation_routes_to_help() -> None:
    route = route_args([])

    assert route.route_kind == "owned"
    assert route.owned_command == "help"


def test_reserved_command_routes_to_owned_handler() -> None:
    route = route_args(["version"])

    assert route.route_kind == "owned"
    assert route.owned_command == "version"


def test_help_flag_is_reserved_for_owned_help() -> None:
    route = route_args(["--help"])

    assert route.route_kind == "owned"
    assert route.owned_command == "help"


def test_unknown_command_routes_to_passthrough_without_reordering() -> None:
    args = ["runs", "ls", "--json"]

    route = route_args(args)

    assert route.route_kind == "passthrough"
    assert route.delegated_args == args
