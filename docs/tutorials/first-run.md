# First run

This tutorial is a short, non-interactive version of the onboarding flow. It’s for people who already have a bot token and want to bootstrap without the setup wizard.

## 1. Create config

Create `~/.takopi/takopi.toml` with your bot token and chat id:

```toml
[transports.telegram]
bot_token = "..."
chat_id = 123456789
```

## 2. Enable commands

Takopi disables commands unless you enable them in config:

```toml
[transports.telegram.commands]
enabled = true
```

## 3. Run Takopi

```sh
takopi
```

Takopi should now respond to `/start`.

## 4. Next

Learn about workflows and how to configure them.

[Conversation modes →](conversation-modes.md)
