"""
voice/stt.py

WhisperSTT: wraps OpenAI's open-source Whisper model for local, free
speech-to-text. No API key, no network call at inference time — the model
weights download once on first use (cached at ~/.cache/whisper/) and every
run after that is fully offline.

Input:  audio file path, OR a numpy float32 array (e.g. straight from
        sounddevice — see voice/vad.py)
Output: transcribed text string

Deps: openai-whisper (pip), ffmpeg (system binary, NOT pip-installable —
      see requirements.txt notes and README for install instructions)

Model size trade-off (set via WHISPER_MODEL_SIZE, default "base"):
    tiny   ~75MB   fastest, lowest accuracy
    base   ~150MB  good balance for a demo (default here)
    small  ~500MB  noticeably better accuracy, slower
    medium ~1.5GB  diminishing returns for English on CPU
"""

import logging
import numpy as np
from typing import Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE = "base"
WHISPER_SAMPLE_RATE = 16000  # Whisper always expects 16kHz mono audio


class WhisperSTT:
    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE, language: Optional[str] = None):
        """
        Args:
            model_size: one of tiny/base/small/medium/large/turbo
            language: force a language code (e.g. "en") to skip language
                      detection and slightly speed up/improve transcription.
                      None = auto-detect (Whisper is multilingual by default).
        """
        # Imported lazily so the rest of the project can be imported/tested
        # without requiring `openai-whisper` + its heavy torch dependency
        # to be installed (e.g. if someone only wants to run text chat).
        import whisper

        self.model_size = model_size
        self.language = language
        logger.info("Loading Whisper model '%s' (first run downloads weights)...", model_size)
        self.model = whisper.load_model(model_size)
        logger.info("Whisper model '%s' loaded.", model_size)

    def transcribe(self, audio: Union[str, np.ndarray]) -> str:
        """
        Transcribes either a file path (str) or an in-memory float32 numpy
        array of audio samples at WHISPER_SAMPLE_RATE (16kHz, mono) — the
        latter is what voice/pipeline.py passes directly from VAD-captured
        microphone audio, avoiding a round-trip through a temp .wav file.

        Returns the transcribed text, stripped of leading/trailing whitespace.
        """
        if isinstance(audio, np.ndarray):
            # Whisper's transcribe() accepts a numpy float32 array directly,
            # as long as it's mono and normalized to [-1, 1] at 16kHz.
            audio_input = audio.astype(np.float32)
        else:
            audio_input = audio  # file path string — Whisper + ffmpeg handle decoding

        options = {}
        if self.language:
            options["language"] = self.language

        try:
            result = self.model.transcribe(audio_input, **options)
        except Exception as e:
            logger.error("Whisper transcription failed: %s", e)
            return ""

        text = result.get("text", "").strip()
        detected_language = result.get("language", "unknown")
        logger.info("Transcribed (%d chars, detected language='%s'): '%s'", len(text), detected_language, text[:80])
        return text
