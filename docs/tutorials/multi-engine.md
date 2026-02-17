# Multi-engine workflows

This tutorial shows you how to use different engines for different tasks and set up defaults so you don't have to think about it.

**What you'll learn:** Engine directives, persistent defaults, and when to use which engine.

## Why multiple engines?

Different engines have different strengths:

| Engine | Good at |
|-------|---------|
| **Codex** | Fast edits, shell commands, quick fixes |
| **Claude Code** | Complex refactors, architecture, long context |
| **OpenCode** | Open-source alternative, local models |
| **Pi** | Conversational, explanations |

You might want Codex for quick tasks and Claude for deep work—without manually specifying every time.

## 1. One-off engine selection

Prefix any message with `/<engine>`:

!!! user "You"
    /claude refactor this module to use dependency injection

!!! user "You"
    /codex add a --verbose flag to the CLI

!!! user "You"
    /pi explain how the event loop works in this codebase

The engine only applies to that message. The response will have a resume line for that engine:

!!! takopi "Takopi"
    💪 · done · claude · 8s<br>
    claude --resume abc123

When you reply, Takopi sees `claude --resume` and automatically uses Claude—you don't need to repeat `/claude`.

## 2. Engine + project + branch

Directives combine. Order doesn't matter:

!!! user "You"
    /claude /happy-gadgets @feat/di refactor to use dependency injection

Or:

!!! user "You"
    /happy-gadgets @feat/di /claude refactor to use dependency injection

Both do the same thing: run Claude in the `happy-gadgets` project on the `feat/di` branch.

!!! note "Directives are only parsed at the start"
    Everything after the first non-directive word is the prompt. `/claude fix /this/path` uses Claude with prompt "fix /this/path"—it doesn't try to parse `/this` as a directive.

## 3. Set a global default engine

Set a default engine in config so new chats use it automatically:

=== "takopi config"

    ```sh
    takopi config set default_engine "claude"
    ```

=== "toml"

    ```toml
    default_engine = "claude"
    ```

## 4. Per-project defaults

Set a default engine in your project config:

=== "takopi config"

    ```sh
    takopi config set projects.happy-gadgets.path "~/dev/happy-gadgets"
    takopi config set projects.happy-gadgets.default_engine "claude"
    ```

=== "toml"

    ```toml
    [projects.happy-gadgets]
    path = "~/dev/happy-gadgets"
    default_engine = "claude"
    ```

Now `/happy-gadgets` tasks default to Claude, even if your global default is Codex.

## 5. Selection precedence

When Takopi picks an engine, it checks (highest to lowest):

1. **Resume line** — replying to `claude --resume ...` uses Claude
2. **Explicit directive** — `/codex ...` uses Codex
3. **Project default** — `default_engine` in project config
4. **Global default** — `default_engine` at the top of `takopi.toml`

This means: resume lines always win, then explicit directives, then the most specific default applies.

!!! note
    With `session_mode = "chat"`, stored sessions are per engine. Replying to a resume line for another engine runs that engine and updates its stored session without overwriting other engines.

!!! example
    Chat sessions with two engines (assume default engine is `codex`):

    1. You send: `fix the failing tests` -> bot replies with `codex resume A` (stores Codex session A).
    2. You reply to an older Claude message containing `claude --resume B` -> runs Claude and stores Claude session B.
    3. You send a new message (not a reply) -> auto-resumes Codex session A (default engine), Claude session B remains stored for future replies or defaults.

## 6. Practical patterns

**Pattern: Quick questions vs. deep work**

=== "takopi config"

    ```sh
    # Global default for quick stuff
    takopi config set default_engine "codex"

    # Project default for complex codebase
    takopi config set projects.backend.path "~/dev/backend"
    takopi config set projects.backend.default_engine "claude"
    ```

=== "toml"

    ```toml
    # Global default for quick stuff
    default_engine = "codex"

    # Project default for complex codebase
    [projects.backend]
    path = "~/dev/backend"
    default_engine = "claude"
    ```

Simple messages go to Codex. `/backend` messages go to Claude.

**Pattern: Override for specific tasks**

Even with defaults, you can always override:

!!! user "You"
    /codex just add a print statement here

Works regardless of what the default is.

## Recap

| Want to... | Do this |
|------------|---------|
| Use an engine once | `/claude ...` or `/codex ...` |
| Set default for project | `default_engine = "..."` in config |
| Set global default | `default_engine = "..."` at top of config |

## You're done!

That's the end of the tutorials. You now know how to:

- ✅ Install and configure Takopi
- ✅ Send tasks and continue conversations
- ✅ Cancel runs mid-flight
- ✅ Target repos and branches from chat
- ✅ Use multiple engines effectively

## Where to go next

**Want to do something specific?**

- [Enable forum topics](../how-to/topics.md) for organized threads
- [Transfer files](../how-to/file-transfer.md) between Telegram and your repo
- [Use voice notes](../how-to/voice-notes.md) to dictate tasks
- [Schedule tasks](../how-to/schedule-tasks.md) to run later

**Want to understand the internals?**

- [Architecture](../explanation/architecture.md) — how the pieces fit together
- [Routing and sessions](../explanation/routing-and-sessions.md) — how context resolution works
- [Specification](../reference/specification.md) — normative behavior contracts

**Need exact syntax?**

- [Commands & directives](../reference/commands-and-directives.md)
- [Configuration](../reference/config.md)
