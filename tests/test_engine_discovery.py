from typing import cast

import pytest

import click
import typer

from takopi import cli, engines, plugins
from takopi.engine_aliases import INTERNAL_ENGINE_ID
from tests.plugin_fixtures import FakeEntryPoint, install_entrypoints


@pytest.fixture
def engine_entrypoints(monkeypatch):
    entrypoints = [
        FakeEntryPoint(
            "codex",
            "takopi.runners.codex:BACKEND",
            plugins.ENGINE_GROUP,
        ),
        FakeEntryPoint(
            "claude",
            "takopi.runners.claude:BACKEND",
            plugins.ENGINE_GROUP,
        ),
        FakeEntryPoint(
            "bad-id",
            "takopi.runners.bad:BACKEND",
            plugins.ENGINE_GROUP,
        ),
    ]
    install_entrypoints(monkeypatch, entrypoints)
    monkeypatch.setattr(cli, "_load_settings_optional", lambda: (None, None))
    return entrypoints


def test_engine_discovery_filters_invalid_ids(engine_entrypoints) -> None:
    ids = engines.list_backend_ids()
    assert ids == ["claude", "codex"]


def test_cli_registers_engine_commands_sorted(engine_entrypoints) -> None:
    app = cli.create_app()
    command_names = [cmd.name for cmd in app.registered_commands]
    engine_ids = engines.list_backend_ids()
    assert INTERNAL_ENGINE_ID in engine_ids
    assert INTERNAL_ENGINE_ID in command_names
    non_internal_ids = [engine_id for engine_id in engine_ids if engine_id != INTERNAL_ENGINE_ID]
    assert not any(engine_id in command_names for engine_id in non_internal_ids)


def test_engine_commands_do_not_expose_engine_id_option(
    engine_entrypoints,
) -> None:
    app = cli.create_app()
    group = cast(click.Group, typer.main.get_command(app))
    engine_ids = [INTERNAL_ENGINE_ID]

    ctx = group.make_context("takopi", [])

    for engine_id in engine_ids:
        command = group.get_command(ctx, engine_id)
        assert command is not None
        options: set[str] = set()
        for param in command.params:
            options.update(getattr(param, "opts", []))
            options.update(getattr(param, "secondary_opts", []))
        assert "--final-notify" in options
        assert "--debug" in options
        assert not any(opt.lstrip("-") == "engine-id" for opt in options)
