"""
voice/tts.py

TTSEngine: wraps pyttsx3 for local, free, offline text-to-speech.
No API key, no network call, works fully offline once installed —
uses the OS's built-in speech engine under the hood (SAPI5 on Windows,
NSSpeechSynthesizer on macOS, espeak on Linux).

Trade-off: pyttsx3 sounds noticeably
robotic compared to a cloud TTS service. The free upgrade path is
`edge-tts` (uses Microsoft Edge's online neural voices, free but requires
network) — mentioned here as the explicit swap-in point, not implemented,
to keep this build's "zero network at runtime" guarantee intact.

Input:  text string
Output: speaks audio through the system's default output device,
        optionally saves to a .wav file instead of/as well as speaking

IMPLEMENTATION NOTE — why each speak() runs in a subprocess:
pyttsx3 has a well-documented issue where re-initializing the engine
in-process (engine = pyttsx3.init() again) fixes the "only speaks once"
bug on most platforms, but on some Windows + SAPI5 setups even a fresh
pyttsx3.init() in the SAME process still shares underlying COM/SAPI5
driver state that doesn't fully reset, so later calls silently no-op or
return from runAndWait() without actually blocking until audio finishes.
Running each utterance in a brand-new OS process guarantees a completely
clean COM/SAPI5 state every single time — there is no in-process state
left to get confused, on any platform. This is slightly heavier (spawns
a short-lived Python process per utterance) but is the most reliable
fix available without replacing pyttsx3 entirely.
"""

import logging
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_RATE = 175    # words per minute, pyttsx3 default is ~200 (a bit fast for clarity)
DEFAULT_VOLUME = 1.0

# This script is passed to a fresh `python -c` subprocess for every single
# utterance — see speak()'s docstring for why. Kept as a plain string
# template (not a separate .py file) so TTSEngine has no extra file
# dependency and nothing else in the project needs to know this exists.
_SPEAK_SCRIPT = """
import sys
import pyttsx3
text = sys.argv[1]
rate = int(sys.argv[2])
volume = float(sys.argv[3])
voice_index = sys.argv[4]
engine = pyttsx3.init()
engine.setProperty("rate", rate)
engine.setProperty("volume", volume)
if voice_index != "":
    voices = engine.getProperty("voices")
    idx = int(voice_index)
    if 0 <= idx < len(voices):
        engine.setProperty("voice", voices[idx].id)
engine.say(text)
engine.runAndWait()
engine.stop()
"""

_SAVE_SCRIPT = """
import sys
import pyttsx3
text = sys.argv[1]
output_path = sys.argv[2]
rate = int(sys.argv[3])
volume = float(sys.argv[4])
voice_index = sys.argv[5]
engine = pyttsx3.init()
engine.setProperty("rate", rate)
engine.setProperty("volume", volume)
if voice_index != "":
    voices = engine.getProperty("voices")
    idx = int(voice_index)
    if 0 <= idx < len(voices):
        engine.setProperty("voice", voices[idx].id)
engine.save_to_file(text, output_path)
engine.runAndWait()
engine.stop()
"""


class TTSEngine:
    def __init__(self, rate: int = DEFAULT_RATE, volume: float = DEFAULT_VOLUME, voice_index: Optional[int] = None):
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index

        # Used only by list_voices() — a single, one-off pyttsx3 call in
        # THIS process is fine, since list_voices() is typically called
        # once at startup, not repeatedly in a loop like speak() is.
        import pyttsx3
        self._pyttsx3 = pyttsx3

    def speak(self, text: str) -> None:
        """
        Speaks `text` out loud through the system's default audio output.
        Blocks until done.

        Runs in a fresh subprocess every call (see module docstring) —
        this is what makes the SECOND and every subsequent call in a
        long-running session (e.g. the voice demo loop) reliably produce
        audio, even on Windows setups where re-initializing pyttsx3
        in-process wasn't enough on its own.
        """
        if not text:
            logger.warning("TTSEngine.speak called with empty text; skipping")
            return
        logger.info("Speaking (%d chars): '%s'", len(text), text[:80])

        voice_index_arg = "" if self.voice_index is None else str(self.voice_index)
        result = subprocess.run(
            [sys.executable, "-c", _SPEAK_SCRIPT, text, str(self.rate), str(self.volume), voice_index_arg],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("TTS subprocess failed (exit code %d): %s", result.returncode, result.stderr[-500:])

    def save_to_file(self, text: str, output_path: str) -> str:
        """Saves `text` as a .wav file at `output_path` instead of speaking it aloud. Also runs in its own subprocess, for the same reason as speak()."""
        if not text:
            raise ValueError("Cannot save empty text to audio file")

        voice_index_arg = "" if self.voice_index is None else str(self.voice_index)
        result = subprocess.run(
            [sys.executable, "-c", _SAVE_SCRIPT, text, output_path, str(self.rate), str(self.volume), voice_index_arg],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("TTS save-to-file subprocess failed (exit code %d): %s", result.returncode, result.stderr[-500:])
            raise RuntimeError(f"Failed to save TTS audio to '{output_path}': {result.stderr[-300:]}")

        logger.info("Saved TTS audio to '%s'", output_path)
        return output_path

    def list_voices(self) -> list:
        """Returns available system voices — useful for picking voice_index. One-off call, safe to run in-process."""
        engine = self._pyttsx3.init()
        voices = engine.getProperty("voices")
        result = [{"index": i, "id": v.id, "name": getattr(v, "name", "unknown")} for i, v in enumerate(voices)]
        engine.stop()
        return result
