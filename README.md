# Harold — Voice-Controlled Claude Agent Session Manager

A personal voice interface for spawning and managing Claude Agent SDK sessions hands-free. Speak tasks to create sessions, receive status updates on demand, and get pinged audibly when sessions complete or error.

## Features

- **Push-to-talk activation** — Hold a hotkey to capture audio
- **Intent classification** — Automatically categorize voice commands (spawn, status, input, list, kill)
- **Multi-session management** — Run multiple Claude Agent tasks in parallel, tracked by name
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
- `spawn_session` — Create a new Claude Agent session with a prompt
- `read_status` — Summarize progress of a named session
- `send_input` — Send follow-up text to an active session
- `list_sessions` — Enumerate all active sessions and states
- `kill_session` — Terminate a session

### 3. Session Management
Each session is backed by a `ClaudeSDKClient` instance. Sessions track:
- Name (auto-inferred from the initial prompt)
- State (running, awaiting_followup, complete, error)
- Full message log (all SDK messages for replay/summarization)
- Background task that consumes `receive_response()` from the SDK

When a session reaches a `ResultMessage`, it transitions to complete/error and triggers a priority ping.

### 4. Status Summaries
Request session status via voice. A lightweight Haiku call summarizes the message log (only new messages since last read) and speaks the summary back to you.

### 5. Priority Pings
When any session finishes or errors, a local audio ping plays immediately (distinct tones for each state) — no server round-trip needed.

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

Configure your API keys and audio device in `config.py`, plus your project-to-path mappings for CWD resolution.

## Usage

Start Harold:
```bash
uv run python main.py
```

Once running:
- **Hold your hotkey** to start recording
- **Release** to send your voice command
- Speak commands like:
  - "Spawn a session to refactor the auth module"
  - "Read status of auth-refactor"
  - "Send: add error handling for edge cases"
  - "List all sessions"
  - "Kill auth-refactor"
- **Listen for audio feedback** and priority pings

## Architecture

```
harold/
├── main.py                  # entry point
├── audio/
│   ├── listener.py          # mic capture + VAD + whisper
│   └── speaker.py           # ElevenLabs TTS + playback
├── router/
│   └── agent.py             # intent classification
├── sessions/
│   ├── manager.py           # session registry & lifecycle
│   └── summarizer.py        # message log summarization
├── config.py                # API keys, settings, project map
└── pyproject.toml
```

### Key Components

**Audio Loop** — Push-to-talk hotkey triggers mic capture. Silero-VAD trims silence, faster-whisper transcribes, text is queued to the router via janus queues (bridge between sync audio thread and async event loop).

**Router Agent** — Stateless Claude call with injected session registry. Classifies intent and returns structured JSON actions.

**Session Manager** — Owns all active sessions. Each session is a `ClaudeSDKClient` instance with a background task consuming `receive_response()`. Message log is appended-only for replay/summarization.

**Summarizer** — Lightweight Haiku call that summarizes unsummarized messages in a session's log (only new content since last read).

**Priority Ping** — Plays local audio tones (distinct for complete vs. error) when sessions finish, bypassing router and TTS.

## Development Phases

1. **Audio skeleton** — mic → VAD → whisper → playback
2. **SDK validation** — standalone script to validate `ClaudeSDKClient` message flow
3. **Single session** — wire SDK into voice pipeline
4. **Router agent** — add intent classification and multi-session support
5. **Priority pings** — completion/error detection and audio feedback
6. **Polish** — session naming, better prompts, error handling

## Notes

- Sessions are in-memory only (lost on restart)
- CWD is resolved via a hardcoded project name → path map in config
- Tool permission mode is auto-accept for the POC (can be made interactive later)
- No session persistence or history between runs
