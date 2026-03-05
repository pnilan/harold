# Harold — Voice-Controlled Claude Agent Session Manager

A personal voice interface for spawning and managing Claude Agent SDK sessions hands-free. Speak tasks to create sessions, receive status updates on demand, send follow-up instructions, and get pinged audibly when sessions complete or error.

## Features

- **Push-to-talk activation** — Hold a hotkey to capture audio
- **Intent classification** — Automatically categorize voice commands (spawn, status, input, list, kill)
- **Multi-session management** — Run multiple Claude Agent tasks in parallel, tracked by name
- **Multi-turn sessions** — Sessions stay alive after completing a turn, accepting follow-up instructions
- **Per-project CWD routing** — Route sessions to the correct working directory based on project name
- **Async status updates** — Request a summary of any session's progress via voice
- **Priority audio pings** — Distinct tones for session completion vs. errors
- **Local VAD & speech recognition** — faster-whisper for transcription (no server-side speech API)
- **Text-to-speech playback** — ElevenLabs for spoken feedback

## How It Works

### 1. Audio Capture
Hold a hotkey to record audio. On release, the captured audio is passed through:
- **Silero-VAD** — trims silence
- **faster-whisper** — transcribes to text
- Text is sent to the router for intent classification

### 2. Intent Router
A stateless Claude API call (Haiku) receives your transcription and routes it to one of:
- `spawn_session` — Create a new Claude Agent session with a prompt (optionally targeting a project)
- `read_status` — Summarize progress of a named session
- `send_input` — Send follow-up text to a session awaiting input
- `list_sessions` — Enumerate all active sessions and states
- `kill_session` — Terminate a session

### 3. Session Management
Each session is backed by a `ClaudeSDKClient` instance. Sessions track:
- Name (auto-inferred from the initial prompt)
- State (`running`, `awaiting_input`, `error`)
- Full message log (all SDK messages for replay/summarization)
- Background task that consumes `receive_response()` from the SDK

When a session reaches a `ResultMessage`, it transitions to `awaiting_input` (success) or `error`, and triggers a priority ping. Sessions in `awaiting_input` stay alive for follow-up instructions.

### 4. Status Summaries
Request session status via voice. A lightweight Haiku call summarizes the message log (only new messages since last read) and speaks the summary back to you.

### 5. Priority Pings
When any session finishes or errors, a local audio ping plays immediately (distinct tones for each state) — no server round-trip needed. A shared playback lock prevents audio collisions between TTS and pings.

### 6. Project CWD Routing
Configure project name-to-path mappings via `HAROLD_PROJECT_PATHS`. When spawning a session, mention a project name and the router will extract it, resolving the correct working directory for the SDK session.

## Stack

| Layer              | Tool                              |
|--------------------|-----------------------------------|
| Mic input/playback | `sounddevice`                     |
| Voice Activity Detection | `silero-VAD`                 |
| Speech-to-text     | `faster-whisper` (local)          |
| Router agent       | Anthropic API (Claude Haiku)      |
| Session management | `claude-agent-sdk` (Python)       |
| Summarization      | Anthropic API (Claude Haiku)      |
| Text-to-speech     | ElevenLabs API                    |

## Installation

### Requirements
- Python 3.10+
- API keys: `ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY`
- Hotkey support (OS-level hotkey binding)

### Setup
```bash
cd harold
uv sync
```

Copy `.env.example` to `.env` and fill in your API keys and settings:

```bash
cp .env.example .env
```

Key environment variables:
- `ANTHROPIC_API_KEY` — Required for router, summarizer, and SDK sessions
- `ELEVENLABS_API_KEY` — Required for TTS playback
- `HAROLD_DEFAULT_CWD` — Default working directory for sessions
- `HAROLD_PROJECT_PATHS` — Comma-separated `name=path` pairs (e.g., `harold=/Users/me/harold,webapp=/Users/me/webapp`)
- `HAROLD_LOG_LEVEL` — Logging level (default: `INFO`)
- `HAROLD_SESSION_MODEL` / `HAROLD_ROUTER_MODEL` / `HAROLD_SUMMARIZER_MODEL` — Override default models

## Usage

Start Harold:
```bash
uv run harold
# or:
uv run python main.py
```

Once running:
- **Hold your hotkey** (F5 by default) to start recording
- **Release** to send your voice command
- Speak commands like:
  - "Start a session on harold to fix the config parser"
  - "Fix the auth bug in the login module"
  - "What's the status of auth-refactor?"
  - "Tell auth-refactor to also add error handling"
  - "List all sessions"
  - "Kill auth-refactor"
- **Listen for audio feedback** and priority pings

## Architecture

```
harold/
├── main.py                  # entry point & main event loop
├── audio/
│   ├── _lock.py             # shared playback lock (threading.Lock)
│   ├── listener.py          # mic capture + VAD + whisper
│   ├── ping.py              # completion/error audio tones
│   └── speaker.py           # ElevenLabs TTS + playback
├── router/
│   ├── agent.py             # intent classification (Haiku)
│   └── models.py            # Pydantic models for router output
├── sessions/
│   ├── manager.py           # session registry & lifecycle
│   └── summarizer.py        # message log summarization (Haiku)
├── config.py                # settings, env var parsing, project paths
└── scripts/
    ├── validate_sdk.py      # SDK message flow validation
    └── validate_router.py   # router intent classification tests
```

### Key Components

**Audio Loop** — Push-to-talk hotkey triggers mic capture. Silero-VAD trims silence, faster-whisper transcribes, text is queued to the router via janus queues (bridge between sync audio thread and async event loop).

**Router Agent** — Stateless Claude call with injected session registry and project list. Classifies intent and returns structured JSON actions.

**Session Manager** — Owns all active sessions. Each session is a `ClaudeSDKClient` instance with a background task consuming `receive_response()`. Sessions transition through `running` → `awaiting_input` (on success) or `error`, with stale session reaping after 5 minutes of idle time.

**Summarizer** — Lightweight Haiku call that summarizes unsummarized messages in a session's log (only new content since last read).

**Priority Ping** — Plays local audio tones (distinct for complete vs. error) when sessions finish, bypassing router and TTS. A shared lock serializes all audio playback to prevent collisions.

## Development Phases

1. **Phase 1: Audio skeleton** — mic → VAD → whisper → playback (COMPLETE)
2. **Phase 2: SDK session manager** — Validate ClaudeSDKClient, single session, summarizer, priority pings (COMPLETE)
3. **Phase 3: Router agent** — Haiku-based intent router, multi-session registry, fuzzy name matching (COMPLETE)
4. **Phase 4: Multi-turn sessions** — send_input intent, AWAITING_INPUT state, sessions stay alive between turns (COMPLETE)
5. **Phase 5: Project CWD routing, error handling & polish** — per-project CWD routing, audio concurrency lock, error handling, logging (COMPLETE)

## Notes

- Sessions are in-memory only (lost on restart)
- CWD is resolved via `HAROLD_PROJECT_PATHS` env var → `HAROLD_DEFAULT_CWD` fallback → process CWD
- Tool permission mode uses `acceptEdits` with a denylist for destructive bash patterns
- No session persistence or history between runs
- Router and summarizer are independent Haiku calls; the router uses SDK default retries (2) for low latency, the summarizer uses 3 retries since it's off the hot path
