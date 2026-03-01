import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from pynput import keyboard
from silero_vad import get_speech_timestamps, load_silero_vad

from harold.config import (
    BLOCKSIZE,
    CHANNELS,
    PUSH_TO_TALK_KEY,
    SAMPLE_RATE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
)


class AudioListener:
    def __init__(
        self,
        on_transcript: Callable[[str], None],
        on_recording_start: Callable[[], None] | None = None,
    ):
        self.on_transcript = on_transcript
        self.on_recording_start = on_recording_start
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._recording = False
        self._lock = threading.Lock()

        # Load models once
        print("Loading Whisper model...")
        self._whisper = WhisperModel(
            WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
        )
        print("Loading Silero VAD model...")
        self._vad = load_silero_vad()
        print("Audio listener ready.")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"Audio status: {status}")
        self._chunks.append(indata.copy())

    def _start_recording(self):
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._chunks = []

        if self.on_recording_start:
            self.on_recording_start()

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCKSIZE,
            callback=self._audio_callback,
        )
        self._stream.start()

    def _stop_recording(self):
        with self._lock:
            if not self._recording:
                return
            self._recording = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._chunks:
            return

        # Process in a separate thread to not block the key listener
        chunks = self._chunks
        self._chunks = []
        threading.Thread(target=self._process_audio, args=(chunks,), daemon=True).start()

    def _process_audio(self, chunks: list[np.ndarray]):
        audio = np.concatenate(chunks, axis=0)  # shape: (N, 1)
        audio_1d = audio.flatten()

        # VAD: trim silence
        audio_tensor = torch.from_numpy(audio_1d)
        timestamps = get_speech_timestamps(audio_tensor, self._vad, sampling_rate=SAMPLE_RATE)

        if not timestamps:
            print("No speech detected.")
            return

        # Extract speech segments and concatenate
        speech_parts = [audio_1d[ts["start"]:ts["end"]] for ts in timestamps]
        speech_audio = np.concatenate(speech_parts)

        # Transcribe
        segments, _info = self._whisper.transcribe(
            speech_audio,
            beam_size=5,
            language="en",
        )
        text = " ".join(seg.text.strip() for seg in segments)

        if text:
            self.on_transcript(text)

    def _on_press(self, key):
        if key == PUSH_TO_TALK_KEY:
            self._start_recording()

    def _on_release(self, key):
        if key == PUSH_TO_TALK_KEY:
            self._stop_recording()

    def start(self):
        self._key_listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._key_listener.start()
        print(f"Push-to-talk active. Hold {PUSH_TO_TALK_KEY} to speak.")

    def stop(self):
        if hasattr(self, "_key_listener"):
            self._key_listener.stop()
        if self._stream:
            self._stream.stop()
            self._stream.close()
