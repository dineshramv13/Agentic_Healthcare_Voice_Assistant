"""
voice/vad.py

VAD: simple energy-threshold voice activity detection. Listens to the
microphone continuously, detects when the user starts speaking (energy
rises above a threshold), and returns the captured audio segment once
the user stops speaking (energy stays below threshold for a configurable
silence window).

This is the pragmatic version mentioned in the architecture review — real
production systems would use Silero VAD or WebRTC VAD (ML-based, more
robust to background noise). Energy-threshold VAD is good enough to demo
and clearly shows understanding of the concept.

Input:  none (listens to the default microphone device)
Output: numpy float32 array of the captured speech segment, at
        voice.stt.WHISPER_SAMPLE_RATE (16kHz mono) — ready to pass
        straight into WhisperSTT.transcribe()

Deps: sounddevice, numpy
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000          # matches Whisper's expected input rate
BLOCK_DURATION_SEC = 0.1     # process audio in 100ms chunks
ENERGY_THRESHOLD = 0.015     # RMS energy above this = "speech detected"
SILENCE_DURATION_SEC = 1.0   # how long energy must stay below threshold to end a turn
MAX_RECORDING_SEC = 30.0     # hard cap so a stuck-open mic doesn't record forever
PRE_SPEECH_BUFFER_BLOCKS = 3 # keep a few blocks before speech starts, so we don't clip the first word


def _rms_energy(block: np.ndarray) -> float:
    """Root-mean-square energy of an audio block — simple, fast loudness measure."""
    return float(np.sqrt(np.mean(np.square(block))))


class VAD:
    """
    Usage:
        vad = VAD()
        print("Listening...")
        audio = vad.listen_for_utterance()   # blocks until user finishes speaking
        # audio is a numpy float32 array, ready for WhisperSTT.transcribe(audio)
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        energy_threshold: float = ENERGY_THRESHOLD,
        silence_duration_sec: float = SILENCE_DURATION_SEC,
        max_recording_sec: float = MAX_RECORDING_SEC,
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_duration_sec = silence_duration_sec
        self.max_recording_sec = max_recording_sec
        self.block_size = int(BLOCK_DURATION_SEC * sample_rate)

    def listen_for_utterance(self) -> np.ndarray:
        """
        Blocks until the user speaks and then stops speaking (or
        max_recording_sec is hit). Returns the captured speech as a
        single concatenated float32 numpy array.
        """
        import sounddevice as sd

        blocks_recorded = []
        pre_speech_buffer = []
        speech_started = False
        silence_blocks_needed = int(self.silence_duration_sec / BLOCK_DURATION_SEC)
        silent_block_count = 0
        max_blocks = int(self.max_recording_sec / BLOCK_DURATION_SEC)
        total_blocks = 0

        logger.info("VAD listening (energy_threshold=%.4f)...", self.energy_threshold)

        with sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32", blocksize=self.block_size
        ) as stream:
            while total_blocks < max_blocks:
                block, overflowed = stream.read(self.block_size)
                if overflowed:
                    logger.warning("Audio input overflowed — some samples may have been dropped")

                block = block.flatten()
                energy = _rms_energy(block)
                total_blocks += 1

                if not speech_started:
                    # Keep a small rolling buffer so we don't clip the very
                    # start of the utterance once speech IS detected.
                    pre_speech_buffer.append(block)
                    if len(pre_speech_buffer) > PRE_SPEECH_BUFFER_BLOCKS:
                        pre_speech_buffer.pop(0)

                    if energy > self.energy_threshold:
                        speech_started = True
                        blocks_recorded.extend(pre_speech_buffer)
                        blocks_recorded.append(block)
                        logger.info("Speech detected (energy=%.4f), recording started", energy)
                else:
                    blocks_recorded.append(block)
                    if energy <= self.energy_threshold:
                        silent_block_count += 1
                        if silent_block_count >= silence_blocks_needed:
                            logger.info("Silence detected — ending utterance capture")
                            break
                    else:
                        silent_block_count = 0  # reset on any renewed speech

        if not blocks_recorded:
            logger.warning("No speech detected within max_recording_sec=%.1f", self.max_recording_sec)
            return np.array([], dtype=np.float32)

        return np.concatenate(blocks_recorded).astype(np.float32)
