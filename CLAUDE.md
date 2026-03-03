# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Harold is a voice-controlled Claude Agent SDK session manager. It uses push-to-talk hotkeys to spawn, manage, and monitor multiple concurrent Claude Agent sessions, with intent classification (spawn/status/input/list/kill) and audio feedback (priority pings for completion/errors).

## Quick Start

Install dependencies:
```bash
uv sync
```

Run the application:
```bash
uv run python main.py
# or directly:
uv run harold
```

Run SDK validation script (tests `ClaudeSDKClient` message flow):
```bash
uv run python scripts/validate_sdk.py
```

## Development Workflow

- **Always create a feature branch** for new features/phases (never commit directly to main)
- When complete, push the feature branch to origin
- User will handle merging to main
- **Never add Claude as co-author on commits or PRs** — Do not include `Co-Authored-By: Claude` lines

## Architecture

### Core Components

**Audio Layer** (`audio/`)
- `listener.py` — Hotkey-triggered mic capture, Silero-VAD for silence trimming, faster-whisper for speech-to-text
- `speaker.py` — ElevenLabs TTS and audio playback
- `ping.py` — Local audio tones for completion/error feedback (no server round-trip)

**Router Agent** (`router/agent.py`)
- Stateless Claude Haiku API call that classifies user intent (spawn/status/list/kill/send_input)
- Uses forced tool_use for structured output; receives injected session registry

**Session Management** (`sessions/`)
- `manager.py` — Multi-session registry, each backed by `ClaudeSDKClient` instance
- Tracks: name (auto-inferred), state (running/awaiting_input/error), message log
- Background task consumes `receive_response()` from SDK
- Transitions to complete/error on `ResultMessage`; triggers priority ping
- `summarizer.py` — Lightweight Haiku call that summarizes new messages in session log (since last read) for status updates

**Configuration** (`config.py`)
- API keys (ANTHROPIC_API_KEY, ELEVENLABS_API_KEY)
- Audio device settings
- Project name → path mapping for CWD resolution

### Data Flow

1. **Audio Capture** — Hold hotkey → mic → Silero-VAD → faster-whisper → text queued to router (via janus queue)
2. **Intent Classification** — Router (Haiku) receives text, injects session registry, returns action JSON
3. **Session Lifecycle** — ClaudeSDKClient spawned/resumed; background task consumes streaming responses
4. **Status Summarization** — On-demand Haiku call summarizes new messages; spoken feedback via TTS
5. **Completion Ping** — LocalResultMessage triggers audio tone (no API call needed)

## Key Technical Decisions

- **In-memory sessions only** — Sessions lost on restart; no persistence between runs
- **Local VAD & speech recognition** — faster-whisper runs locally (no server-side speech API)
- **Janus queues** — Bridge between sync audio thread and async event loop
- **Priority pings** — Local audio tones bypass router/TTS for immediate feedback
- **CWD resolution** — Hardcoded project name → path map in config (can be extended)
- **Tool permission mode** — Auto-accept for POC (interactive mode can be added later)

## Development Phases

The project is organized in phases (tracked as feature branches):

1. **Phase 1: Audio skeleton** — mic → VAD → whisper → playback (COMPLETE)
2. **Phase 2: SDK session manager** — Validate ClaudeSDKClient, single session, summarizer, priority pings (COMPLETE)
3. **Phase 3: Router agent** — Haiku-based intent router (spawn/status/list/kill), multi-session registry, fuzzy name matching (COMPLETE)
4. **Phase 4: Multi-turn sessions** — send_input intent, AWAITING_INPUT state, sessions stay alive between turns (IN PROGRESS)
5. **Phase 5+** — project CWD routing, polish, error handling

Each phase is merged via PR; check git log for completed phases.

## Environment & Dependencies

- Python 3.10+
- Key packages:
  - `anthropic` — Anthropic API (router & summarizer)
  - `claude-agent-sdk` — Session management
  - `elevenlabs` — TTS playback
  - `faster-whisper` — Local speech-to-text
  - `silero-vad` — Voice activity detection
  - `sounddevice` — Audio I/O
  - `pynput` — Hotkey binding
  - `janus` — Sync/async queue bridge

API keys required: `ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY` (set in `.env`)

## Notes for Future Development

- Router and summarizer are independent Haiku calls; consider batching if latency becomes an issue
- Session message logs are appended-only; consider memory implications if sessions run very long
- Hotkey binding uses OS-level support; may vary by platform
- Tool auto-accept mode should be made interactive once UX is stable
