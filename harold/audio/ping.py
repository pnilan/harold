"""Notification tones for session completion and errors.

Generates short chime/tone sounds using numpy and plays them via sounddevice.
All functions are synchronous — call via asyncio.to_thread() from async code.
"""

import logging

import numpy as np
import sounddevice as sd

from harold.audio._lock import playback_lock

logger = logging.getLogger(__name__)

PING_SAMPLE_RATE = 24000


def _make_tone(freq: float, duration: float, volume: float = 0.4) -> np.ndarray:
    """Generate a sine tone with fade-in/fade-out envelope."""
    t = np.linspace(0, duration, int(PING_SAMPLE_RATE * duration), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t) * volume

    # 10ms fade in/out to prevent clicks
    fade_samples = int(PING_SAMPLE_RATE * 0.01)
    tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
    tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    return tone.astype(np.float32)


def play_complete_ping() -> None:
    """Ascending two-note chime (C5 -> E5), ~350ms total."""
    try:
        c5 = _make_tone(523.25, 0.15)
        e5 = _make_tone(659.25, 0.20)
        silence = np.zeros(int(PING_SAMPLE_RATE * 0.02), dtype=np.float32)

        audio = np.concatenate([c5, silence, e5])
        with playback_lock:
            sd.play(audio, samplerate=PING_SAMPLE_RATE)
            sd.wait()
    except Exception:
        logger.exception("Complete ping failed")


def play_error_ping() -> None:
    """Descending two-note tone (A4 -> E4), ~350ms total."""
    try:
        a4 = _make_tone(440.0, 0.15)
        e4 = _make_tone(329.63, 0.20)
        silence = np.zeros(int(PING_SAMPLE_RATE * 0.02), dtype=np.float32)

        audio = np.concatenate([a4, silence, e4])
        with playback_lock:
            sd.play(audio, samplerate=PING_SAMPLE_RATE)
            sd.wait()
    except Exception:
        logger.exception("Error ping failed")
