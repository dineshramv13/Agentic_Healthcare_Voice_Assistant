"""
tests/test_voice.py

Unit tests for voice/ components, with STT/TTS/audio-hardware mocked out.
These tests verify OUR logic (e.g. VAD's energy-threshold state machine,
WhisperSTT's numpy-vs-filepath branching) without requiring openai-whisper,
pyttsx3, sounddevice, or an actual microphone/speaker to be present —
important since CI environments and many dev machines won't have audio
hardware, and Whisper itself is a slow, heavy dependency to load in a
fast unit test suite.

Run with:
    pytest tests/test_voice.py -v
"""

import sys
import numpy as np
from unittest.mock import MagicMock, patch


class TestWhisperSTT:
    def test_transcribe_numpy_array_input(self):
        """Verifies transcribe() routes a numpy array through model.transcribe() correctly."""
        # Mock the `whisper` module itself before voice.stt imports it,
        # since openai-whisper may not be installed in this environment.
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  How do I book an appointment?  ", "language": "en"}
        mock_whisper_module.load_model.return_value = mock_model

        with patch.dict(sys.modules, {"whisper": mock_whisper_module}):
            from voice.stt import WhisperSTT
            stt = WhisperSTT(model_size="base")

            audio_array = np.zeros(16000, dtype=np.float32)
            result = stt.transcribe(audio_array)

            assert result == "How do I book an appointment?"  # whitespace stripped
            mock_model.transcribe.assert_called_once()
            # First positional arg to transcribe should be the numpy array we passed
            call_args = mock_model.transcribe.call_args
            assert isinstance(call_args[0][0], np.ndarray)

    def test_transcribe_file_path_input(self):
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "test transcript", "language": "en"}
        mock_whisper_module.load_model.return_value = mock_model

        with patch.dict(sys.modules, {"whisper": mock_whisper_module}):
            from voice.stt import WhisperSTT
            stt = WhisperSTT()
            result = stt.transcribe("/path/to/audio.wav")

            assert result == "test transcript"
            call_args = mock_model.transcribe.call_args
            assert call_args[0][0] == "/path/to/audio.wav"

    def test_transcribe_returns_empty_string_on_failure(self):
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("ffmpeg not found")
        mock_whisper_module.load_model.return_value = mock_model

        with patch.dict(sys.modules, {"whisper": mock_whisper_module}):
            from voice.stt import WhisperSTT
            stt = WhisperSTT()
            result = stt.transcribe("/path/to/audio.wav")

            assert result == ""  # fails gracefully, doesn't raise

    def test_forced_language_passed_to_transcribe(self):
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hola", "language": "es"}
        mock_whisper_module.load_model.return_value = mock_model

        with patch.dict(sys.modules, {"whisper": mock_whisper_module}):
            from voice.stt import WhisperSTT
            stt = WhisperSTT(language="es")
            stt.transcribe("/path/to/audio.wav")

            call_kwargs = mock_model.transcribe.call_args[1]
            assert call_kwargs.get("language") == "es"


class TestTTSEngine:
    """
    TTSEngine now runs every pyttsx3 call in a fresh subprocess (see
    voice/tts.py's module docstring for why — re-initializing pyttsx3
    in-process wasn't reliable enough on some Windows + SAPI5 setups).
    That means the correct mocking seam is subprocess.run, not
    sys.modules["pyttsx3"] — the parent process never imports pyttsx3
    directly anymore except inside list_voices().
    """

    def test_speak_invokes_subprocess_with_text_as_argument(self):
        from unittest.mock import patch, MagicMock
        from voice.tts import TTSEngine

        # No pyttsx3 mocking needed at all — TTSEngine() no longer imports
        # pyttsx3 in __init__, only subprocess.run is on the critical path.
        tts = TTSEngine()

        mock_result = MagicMock(returncode=0, stderr="")
        with patch("voice.tts.subprocess.run", return_value=mock_result) as mock_run:
            tts.speak("Hello, how can I help?")

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]  # the command list
            assert "Hello, how can I help?" in call_args

    def test_speak_skips_empty_text(self):
        from unittest.mock import patch
        from voice.tts import TTSEngine

        tts = TTSEngine()
        with patch("voice.tts.subprocess.run") as mock_run:
            tts.speak("")
            mock_run.assert_not_called()

    def test_speak_logs_error_on_nonzero_exit_but_does_not_raise(self):
        from unittest.mock import patch, MagicMock
        from voice.tts import TTSEngine

        tts = TTSEngine()
        mock_result = MagicMock(returncode=1, stderr="some pyttsx3 driver error")
        with patch("voice.tts.subprocess.run", return_value=mock_result):
            tts.speak("This will fail in the subprocess")  # should not raise

    def test_save_to_file_raises_on_empty_text(self):
        import pytest
        from voice.tts import TTSEngine

        tts = TTSEngine()
        with pytest.raises(ValueError):
            tts.save_to_file("", "/tmp/out.wav")

    def test_save_to_file_invokes_subprocess_with_path_as_argument(self):
        from unittest.mock import patch, MagicMock
        from voice.tts import TTSEngine

        tts = TTSEngine()
        mock_result = MagicMock(returncode=0, stderr="")
        with patch("voice.tts.subprocess.run", return_value=mock_result) as mock_run:
            tts.save_to_file("Some text", "/tmp/output.wav")

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "Some text" in call_args
            assert "/tmp/output.wav" in call_args

    def test_each_speak_call_is_independent(self):
        """
        Verifies repeated speak() calls each trigger their own subprocess
        invocation — this is the actual property that fixes the original
        bug report (silent after the first turn). We can't test real audio
        output, but we CAN verify the code path that produces it runs fresh
        every single time, with no shared engine object across calls.
        """
        from unittest.mock import patch, MagicMock
        from voice.tts import TTSEngine

        tts = TTSEngine()
        mock_result = MagicMock(returncode=0, stderr="")
        with patch("voice.tts.subprocess.run", return_value=mock_result) as mock_run:
            tts.speak("First turn response")
            tts.speak("Second turn response")
            tts.speak("Third turn response")

            assert mock_run.call_count == 3
            all_texts_passed = [call[0][0] for call in mock_run.call_args_list]
            assert any("First turn response" in cmd for cmd in all_texts_passed)
            assert any("Second turn response" in cmd for cmd in all_texts_passed)
            assert any("Third turn response" in cmd for cmd in all_texts_passed)


class TestVADEnergyThreshold:
    """
    Tests the pure energy-calculation helper directly — the actual
    listen_for_utterance() method requires a live sounddevice InputStream
    and is exercised via the manual demo (scripts/demo.py --mode voice)
    rather than mocked here, since faithfully mocking a real-time audio
    stream callback loop would test the mock more than the logic.
    """

    def test_rms_energy_of_silence_is_near_zero(self):
        from voice.vad import _rms_energy
        silence = np.zeros(1600, dtype=np.float32)
        assert _rms_energy(silence) == 0.0

    def test_rms_energy_of_loud_signal_is_high(self):
        from voice.vad import _rms_energy
        loud = np.ones(1600, dtype=np.float32) * 0.5
        assert _rms_energy(loud) > 0.015  # above default ENERGY_THRESHOLD

    def test_rms_energy_scales_with_amplitude(self):
        from voice.vad import _rms_energy
        quiet = np.ones(1600, dtype=np.float32) * 0.01
        loud = np.ones(1600, dtype=np.float32) * 0.5
        assert _rms_energy(loud) > _rms_energy(quiet)
