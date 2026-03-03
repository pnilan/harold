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
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID") or "JJQDkHrp6uKU5Vk0WKhY"
ELEVENLABS_MODEL_ID = os.environ.get("ELEVENLABS_MODEL_ID") or "eleven_flash_v2_5"
ELEVENLABS_SAMPLE_RATE = 24000
ELEVENLABS_SPEAKER_BOOST = False
TTS_GAIN: float = float(os.environ.get("HAROLD_TTS_GAIN", "0.5"))

# Claude Agent SDK
# `or None` converts an empty string "" to None (os.environ.get already returns
# None for missing keys, but HAROLD_DEFAULT_CWD="" should also mean "unset").
DEFAULT_CWD: str | None = os.environ.get("HAROLD_DEFAULT_CWD") or None
CLAUDE_MAX_BUDGET_USD: float = 1.00
SUMMARIZER_MODEL: str = os.environ.get("HAROLD_SUMMARIZER_MODEL") or "claude-haiku-4-5-20251001"
ROUTER_MODEL: str = os.environ.get("HAROLD_ROUTER_MODEL") or "claude-haiku-4-5-20251001"
SESSION_MODEL: str = os.environ.get("HAROLD_SESSION_MODEL") or "claude-sonnet-4-20250514"
LOG_LEVEL: str = os.environ.get("HAROLD_LOG_LEVEL", "INFO")

# Project paths (placeholder for Phase 3 router)
PROJECT_PATHS: dict[str, str] = {}
