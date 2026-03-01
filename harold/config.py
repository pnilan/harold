import os

from pynput.keyboard import Key

# Push-to-talk
PUSH_TO_TALK_KEY = Key.f5

# Audio capture
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 512

# Whisper
WHISPER_MODEL = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# ElevenLabs
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
ELEVENLABS_MODEL_ID = "eleven_flash_v2_5"
ELEVENLABS_SAMPLE_RATE = 24000

# Claude Agent SDK
DEFAULT_CWD: str | None = os.environ.get("HAROLD_DEFAULT_CWD") or None
CLAUDE_MAX_BUDGET_USD: float = 1.00
SUMMARIZER_MODEL: str = "claude-haiku-4-5-20251001"
LOG_LEVEL: str = os.environ.get("HAROLD_LOG_LEVEL", "INFO")

# Project paths (placeholder for Phase 3 router)
PROJECT_PATHS: dict[str, str] = {}
