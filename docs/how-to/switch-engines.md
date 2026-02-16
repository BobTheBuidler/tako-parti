# Switch engines

Run a one-off message on a specific engine, or set a persistent default for a chat/topic.

## Use an engine for one message

Prefix the first non-empty line with an engine directive:

```
/codex hard reset the timeline
/claude shrink and store artifacts forever
/opencode hide their paper until they reply
/pi render a diorama of this timeline
```

Directives are only parsed at the start of the first non-empty line.

## Set a default engine

Use config defaults instead of chat commands:

=== "takopi config"

    ```sh
    # global default
    takopi config set default_engine "claude"

    # per-project default
    takopi config set projects.backend.default_engine "claude"
    ```

=== "toml"

    ```toml
    default_engine = "claude"

    [projects.backend]
    default_engine = "claude"
    ```

Selection precedence (highest to lowest): resume token → `/<engine-id>` directive → project default → global default.

## Engine installation

Takopi shells out to engine CLIs. Install them and make sure they’re on your `PATH`
(`codex`, `claude`, `opencode`, `pi`). Authentication is handled by each CLI.

## Related

- [Commands & directives](../reference/commands-and-directives.md)
- [Config reference](../reference/config.md)
