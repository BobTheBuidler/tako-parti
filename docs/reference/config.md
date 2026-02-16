# Configuration

Takopi reads configuration from `~/.takopi/takopi.toml`.

If you expect to edit config while Takopi is running, set:

=== "takopi config"

    ```sh
    takopi config set watch_config true
    ```

=== "toml"

    ```toml
    watch_config = true
    ```

## Top-level keys

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `watch_config` | bool | `false` | Hot-reload config changes (transport excluded). |
| `default_engine` | string | `"codex"` | Default engine id for new threads. |
| `default_project` | string\|null | `null` | Default project alias. |
| `transport` | string | `"telegram"` | Transport backend id. |

## `transports.telegram`

=== "takopi config"

    ```sh
    takopi config set transports.telegram.bot_token "..."
    takopi config set transports.telegram.chat_id 123
    ```

=== "toml"

    ```toml
    [transports.telegram]
    bot_token = "..."
    chat_id = 123
    ```

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `bot_token` | string | (required) | Telegram bot token from @BotFather. |
| `chat_id` | int | (required) | Default chat id. |
| `allowed_user_ids` | int[] | `[]` | Allowed sender user ids. Empty disables sender filtering; when set, only these users can interact (including DMs). |
| `message_overflow` | `"trim"`\|`"split"` | `"trim"` | How to handle long final responses. |
| `forward_coalesce_s` | float | `1.0` | Quiet window for combining a prompt with immediately-following forwarded messages; set `0` to disable. |
| `voice_transcription` | bool | `false` | Enable voice note transcription. |
| `voice_max_bytes` | int | `10485760` | Max voice note size (bytes). |
| `voice_transcription_model` | string | `"gpt-4o-mini-transcribe"` | OpenAI transcription model name. |
| `voice_transcription_base_url` | string\|null | `null` | Override base URL for voice transcription only. |
| `voice_transcription_api_key` | string\|null | `null` | Override API key for voice transcription only. |
| `session_mode` | `"chat"`\|`"stateless"` | `"chat"` | Chat mode auto-resumes; stateless requires reply to continue. |
| `show_resume_line` | bool | `false` | Show resume lines in Telegram responses. |
| `truncate_progress_tool_calls` | bool | `true` | Remove tool-call details from progress messages (cleaner in chat). |
| `truncate_progress_tool_calls_newline` | bool | `true` | Replace tool-call details with a newline.
| `progress_tool_calls_max` | int | `5` | Max number of tool calls to show if truncation is off. |
| `progress_tool_calls_inline` | bool | `false` | Show tool calls inline vs in separate lines. |
| `show_resume_button` | bool | `true` | Show a button on the final message for resuming the conversation. |
| `show_context_footer` | bool | `true` | Show a `ctx: <project> @<branch>` footer line on final messages. |
| `show_progress_header` | bool | `true` | Show header with agent + elapsed time. |
| `progress_header_template` | string | `"🤖 · working · {engine} · {elapsed} · step {step}"` | Template for progress header. |
| `progress_line_template` | string | `"{prefix} {label}"` | Template for progress detail lines. |
| `show_agent_name` | bool | `true` | Show the engine id in the header when `progress_header_template` uses `{engine}`. |
| `show_progress_steps` | bool | `true` | Include the `step N` suffix in the header. |
| `show_progress_tools` | bool | `true` | Show tool-call details when truncation is off. |
| `show_context_header` | bool | `true` | Show a context line above progress.
| `show_context_header_title` | string | `"Context"` | Label for the context header.
| `show_context_header_title_emoji` | string | `"📍"` | Emoji for the context header title.
| `show_context_header_padding` | bool | `true` | Insert a blank line after the context header.
| `show_context_header_project_emoji` | string | `"📁"` | Emoji for the project name line.
| `show_context_header_branch_emoji` | string | `"🌿"` | Emoji for the branch line.
| `show_context_header_style` | `"inline"`\|`"block"` | `"block"` | Render context header in a block or inline style.
| `show_context_header_branch_prefix` | string | `"@"` | Prefix for branch name.
| `show_context_header_project_prefix` | string | `""` | Prefix for project name.
| `show_context_header_project_in_branch` | bool | `true` | Include project in branch line when showing block context header.
| `forward_edits` | bool | `false` | Forward edited progress messages to the chat. |

## `transports.telegram.session_mode`

- `chat` (default): new messages auto-resume the last session.
- `stateless`: you must reply to a resume line to continue a conversation.

## `transports.telegram.topics`

=== "takopi config"

    ```sh
    takopi config set transports.telegram.topics.enabled true
    takopi config set transports.telegram.topics.scope "auto"
    takopi config set transports.telegram.topics.index_prefix "takopi"
    takopi config set transports.telegram.topics.rename_format "{project} {branch}"
    ```

=== "toml"

    ```toml
    [transports.telegram.topics]
    enabled = true
    scope = "auto"
    index_prefix = "takopi"
    rename_format = "{project} {branch}"
    ```

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `enabled` | bool | `false` | Enable topics support. |
| `scope` | `"auto"`\|`"main"`\|`"projects"`\|`"all"` | `"auto"` | Where topics are enabled. |
| `index_prefix` | string | `"takopi"` | Prefix for topic names when auto-creating. |
| `rename_format` | string | `"{project} {branch}"` | Format for renaming topics. |
| `archive_on_unbind` | bool | `true` | Archive topic after clearing a binding.
| `synthesize` | bool | `false` | Create a synthetic topic binding for non-topic chats.

## `transports.telegram.files`

=== "takopi config"

    ```sh
    takopi config set transports.telegram.files.enabled true
    takopi config set transports.telegram.files.auto_put true
    takopi config set transports.telegram.files.auto_put_mode "upload"
    takopi config set transports.telegram.files.uploads_dir "incoming"
    takopi config set transports.telegram.files.allowed_user_ids "[123456789]"
    takopi config set transports.telegram.files.deny_globs '[".git/**", ".env", ".envrc", "**/*.pem", "**/.ssh/**"]'
    ```

=== "toml"

    ```toml
    [transports.telegram.files]
    enabled = true
    auto_put = true
    auto_put_mode = "upload" # upload | prompt
    uploads_dir = "incoming"
    allowed_user_ids = [123456789]
    deny_globs = [".git/**", ".env", ".envrc", "**/*.pem", "**/.ssh/**"]
    ```

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `enabled` | bool | `false` | Enable file transfer. |
| `auto_put` | bool | `false` | Auto-save files without prompting. |
| `auto_put_mode` | `"upload"`\|`"prompt"` | `"prompt"` | File handling behavior. |
| `uploads_dir` | string | `"incoming"` | Directory for uploads.
| `allowed_user_ids` | int[] | `[]` | Allowed senders for file uploads. Empty disables sender filtering; when set, only these users can upload.
| `deny_globs` | string[] | `[]` | Glob patterns that are rejected even when a user is allowed.

## `transports.telegram.notifications`

```toml
[transports.telegram.notifications]
# Try `notify` first; fall back to `mention`.
mode = "notify" # notify | mention
```

## `transports.telegram.voice_transcription`

```toml
[transports.telegram.voice_transcription]
enabled = true
model = "gpt-4o-mini-transcribe"
```

## `transports.telegram.voice_transcription` (direct overrides)

```toml
[transports.telegram.voice_transcription]
api_key = "..." # override base api key
base_url = "https://api.openai.com/v1" # override base api url
```

## `transports.telegram.defaults`

You can set per-chat defaults for:

- `engine`
- `project`
- `branch`

## `transports.telegram.shortcuts`

Define in-chat shortcuts:

```toml
[transports.telegram.shortcuts]
"/prod" = "/backend @main"
```

## `transports.telegram.commands`

=== "takopi config"

    ```sh
    takopi config set transports.telegram.commands.enabled true
    takopi config set transports.telegram.commands.manage_mode "auto"
    takopi config set transports.telegram.commands.sanitize_quotes "trim"
    ```

=== "toml"

    ```toml
    [transports.telegram.commands]
    enabled = true
    manage_mode = "auto" # auto | on | off
    sanitize_quotes = "trim" # trim | strip | none
    ```

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `enabled` | bool | `true` | Enable Telegram commands. |
| `manage_mode` | `"auto"`\|`"on"`\|`"off"` | `"auto"` | Refresh bot commands on startup. |
| `sanitize_quotes` | `"trim"`\|`"strip"`\|`"none"` | `"trim"` | How to sanitize quoted replies passed into the prompt. |

## `transports.telegram.commands` (continued)

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `summary_prompt` | string | `"You are a helpful assistant."` | System prompt for summary command.
| `summary_temperature` | float | `0.2` | Summary temperature.
| `summary_max_tokens` | int | `512` | Summary max tokens.
| `topics_prompt` | string | `"You are a helpful assistant."` | System prompt for topics commands.
| `topics_temperature` | float | `0.2` | Summary temperature.
| `topics_max_tokens` | int | `512` | Summary max tokens.
| `topics_summarize_message_limit` | int | `50` | Max messages to summarize.
| `topics_summarize_latest` | bool | `false` | Summarize only latest message.
| `topics_prompt_style` | `"short"`\|`"long"` | `"short"` | Prompt shape for topics response.
| `topics_reply_style` | `"short"`\|`"long"` | `"short"` | Response shape for topics.
| `topics_progress_placeholder` | string | `"Summarizing…"` | Placeholder while topics summary runs.
| `topics_summary_template` | string | `"{summary}"` | Template for summary output.
| `topics_summary_entry_template` | string | `"- {summary}"` | Template for the bullet entries.
| `topics_summary_none` | string | `"No recent messages."` | Output when no messages.
| `topics_summary_intro` | string | `"Recent messages"` | Title for summary.
| `topics_summary_show_chat_label` | bool | `true` | Show chat label in summary.

## `transports.telegram.commands` (chat sessions)

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `sessions_enabled` | bool | `true` | Store session mappings in memory.
| `sessions_idle_ttl_s` | float | `86400` | Session TTL.
| `sessions_store_path` | string | `"telegram_sessions.json"` | Session store file.
| `session_mode` | `"chat"`\|`"stateless"` | `"chat"` | How the chat handles new messages. |
| `final_resume_line` | `"auto"`\|`"always"`\|`"never"` | `"auto"` | When to show resume lines for chat sessions. |

## `transports.telegram.commands` (allrepos)

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `allrepos_enabled` | bool | `false` | Enable allrepos command.
| `allrepos_default_glob` | string | `"**/*"` | Default file glob.
| `allrepos_auto_generate_glob` | bool | `true` | Auto-apply exclude filters if absent.
| `allrepos_default_exclude` | string | `".git/**"` | Default exclude glob.

## `transports.telegram.commands` (swarm)

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `swarm_enabled` | bool | `false` | Enable swarm command.
| `swarm_branch_prefix` | string | `"swarm"` | Prefix for swarm branches.
| `swarm_branch_length` | int | `8` | Random suffix length.
| `swarm_topic_prefix` | string | `"swarm"` | Prefix for swarm topics.
| `swarm_topic_length` | int | `8` | Random suffix length.

## Related

- [Commands & directives](commands-and-directives.md)
- [Transport: Telegram](transports/telegram.md)
- [Tutorial: file transfer](../how-to/file-transfer.md)
