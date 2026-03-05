import logging
import os

from pynput.keyboard import Key

logger = logging.getLogger(__name__)

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
ROUTER_MAX_RETRIES: int = int(os.environ.get("HAROLD_ROUTER_MAX_RETRIES", "2"))
SUMMARIZER_MAX_RETRIES: int = int(os.environ.get("HAROLD_SUMMARIZER_MAX_RETRIES", "3"))


def _parse_project_paths() -> dict[str, str]:
    """Parse HAROLD_PROJECT_PATHS env var into a {name: path} dict.

    Format: comma-separated name=path pairs.
    Example: HAROLD_PROJECT_PATHS=harold=/Users/me/harold,webapp=/Users/me/webapp

    Keys are normalized to lowercase. Invalid paths are skipped with a warning.

    Limitation: paths containing commas or names containing '=' are not supported.
    Use a config file if you need those characters in paths.
    """
    raw = os.environ.get("HAROLD_PROJECT_PATHS", "").strip()
    if not raw:
        return {}

    paths: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if "=" not in entry:
            logger.warning("Skipping malformed project path entry (no '='): %r", entry)
            continue
        name, _, path = entry.partition("=")
        name = name.strip().lower()
        path = path.strip()
        if not name or not path:
            logger.warning("Skipping empty project path entry: %r", entry)
            continue
        if not os.path.isdir(path):
            logger.warning("Skipping project %r — path does not exist: %s", name, path)
            continue
        paths[name] = path

    logger.info("Loaded project paths: %s", list(paths.keys()) if paths else "(none)")
    return paths


PROJECT_PATHS: dict[str, str] = _parse_project_paths()
