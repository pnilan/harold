import numpy as np
import sounddevice as sd
from elevenlabs.client import ElevenLabs
from elevenlabs.types import VoiceSettings

from harold.config import (
    ELEVENLABS_MODEL_ID,
    ELEVENLABS_SAMPLE_RATE,
    ELEVENLABS_SPEAKER_BOOST,
    ELEVENLABS_VOICE_ID,
    TTS_GAIN,
)


class Speaker:
    def __init__(self, voice_id: str = ELEVENLABS_VOICE_ID):
        self._client = ElevenLabs()
        self._voice_id = voice_id
        self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing

    def speak(self, text: str):
        audio_iter = self._client.text_to_speech.convert(
            text=text,
            voice_id=self._voice_id,
            model_id=ELEVENLABS_MODEL_ID,
            output_format=f"pcm_{ELEVENLABS_SAMPLE_RATE}",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                use_speaker_boost=ELEVENLABS_SPEAKER_BOOST,
            ),
        )
        pcm_data = b"".join(audio_iter)

        if not pcm_data:
            return

        audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
        audio_array = np.clip(audio_array * TTS_GAIN, -32768, 32767).astype(np.int16)
        self._playing = True
        sd.play(audio_array, samplerate=ELEVENLABS_SAMPLE_RATE)
        sd.wait()
        self._playing = False

    def stop(self):
        sd.stop()
        self._playing = False
