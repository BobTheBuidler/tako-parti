# Conversation modes

Takopi can run in two distinct modes: **assistant** and **handoff**.

- **assistant**: ongoing chat (you don't need to reply)
- **handoff**: reply-to-continue (resume from specific messages)

## Assistant mode (default)

Assistant mode is designed for ongoing, conversational work. You send messages normally and Takopi auto-resumes the last engine session.

**Example**

!!! user "You"
    build a new command and wire it into the menu

!!! takopi "Takopi"
    💪 · done · codex · 9s · step 4

!!! user "You"
    tweak the help text

Takopi keeps the same Codex session and resumes from there.

## Handoff mode

Handoff mode requires explicit replies. Takopi shows resume lines by default.

**Example**

!!! takopi "Takopi"
    💪 · done · codex · 7s · step 2<br>
    codex resume 019bb89b-1b0b-7e90-96e4-c33181b49714

!!! user "You"
    [replying to resume line]
    now add tests

## Switching modes

You can switch modes in your config:

=== "takopi config"

    ```sh
    takopi config set transports.telegram.session_mode "handoff"
    takopi config set transports.telegram.show_resume_line true
    ```

=== "toml"

    ```toml
    [transports.telegram]
    session_mode = "stateless" # reply-to-continue
    show_resume_line = true
    ```

## How mode affects answers

| Feature | Assistant | Handoff |
|---------|-----------|---------|
| Auto-resume | ✅ | ❌ |
| Resume lines | ❌ | ✅ |
| Replies required | ❌ | ✅ |

## Related

- [Switch engines](../how-to/switch-engines.md)
- [Commands & directives](../reference/commands-and-directives.md)
