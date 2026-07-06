"""
voice/pipeline.py

VoicePipeline: orchestrates the full voice flow.
    microphone -> VAD captures utterance -> WhisperSTT transcribes
    -> text enters the SAME LangGraph agent used by /chat
    -> response text -> TTSEngine speaks it aloud

This is deliberately built on top of the same `agent/graph.py` graph and
`memory/session.py` session memory as the text path — voice is just a
different "input adapter" and "output adapter" around the identical brain,
the agent doesn't know or care whether the text came from typing or from
Whisper.

Input:  none (uses the microphone) — or a pre-recorded audio file path
        for non-interactive testing (see transcribe_file_to_response)
Output: spoken response (and the text response, returned for logging/display)
"""

import logging
import uuid
from typing import Optional

from agent.graph import get_graph
from memory.session import SessionMemory
from voice.stt import WhisperSTT
from voice.tts import TTSEngine
from voice.vad import VAD

logger = logging.getLogger(__name__)


class VoicePipeline:
    def __init__(
        self,
        stt: Optional[WhisperSTT] = None,
        tts: Optional[TTSEngine] = None,
        vad: Optional[VAD] = None,
        memory: Optional[SessionMemory] = None,
    ):
        # Each of these lazily imports its heavy dependency (whisper,
        # pyttsx3, sounddevice) only when actually instantiated — see the
        # respective files' __init__ methods.
        self.stt = stt or WhisperSTT()
        self.tts = tts or TTSEngine()
        self.vad = vad or VAD()
        self.memory = memory or SessionMemory()
        self.graph = get_graph()

    def _run_agent(self, user_message: str, session_id: str) -> dict:
        """Shared logic for invoking the agent graph + persisting memory — identical to api/server.py's /chat route."""
        trace_id = str(uuid.uuid4())
        history_text = self.memory.get_history_as_text(session_id, last_n=6)

        result = self.graph.invoke({
            "user_message": user_message,
            "session_id": session_id,
            "trace_id": trace_id,
            "conversation_history": history_text,
            "retry_count": 0,
        })

        final_response = result.get("final_response") or result.get("response") or (
            "Sorry, something went wrong and I couldn't generate a response."
        )

        self.memory.add_turn(session_id, role="user", content=user_message, intent=result.get("intent"))
        self.memory.add_turn(session_id, role="assistant", content=final_response)

        return {**result, "final_response": final_response}

    def listen_and_respond(self, session_id: Optional[str] = None) -> dict:
        """
        Full interactive voice turn: listens to the microphone, transcribes,
        runs the agent, and speaks the response aloud.

        Returns a dict with: transcribed_text, response_text, intent, verified.
        """
        session_id = session_id or str(uuid.uuid4())

        print("🎤 Listening...")
        audio = self.vad.listen_for_utterance()

        if audio.size == 0:
            message = "I didn't catch that — could you say it again?"
            self.tts.speak(message)
            return {"transcribed_text": "", "response_text": message, "intent": None, "verified": None}

        transcribed_text = self.stt.transcribe(audio)
        if not transcribed_text:
            message = "Sorry, I couldn't understand that. Could you repeat it?"
            self.tts.speak(message)
            return {"transcribed_text": "", "response_text": message, "intent": None, "verified": None}

        print(f"🗣️  You said: {transcribed_text}")

        result = self._run_agent(transcribed_text, session_id)
        response_text = result["final_response"]

        print(f"🤖 AI: {response_text}")
        self.tts.speak(response_text)

        return {
            "transcribed_text": transcribed_text,
            "response_text": response_text,
            "intent": result.get("intent"),
            "verified": result.get("verified"),
            "session_id": session_id,
        }

    def transcribe_file_to_response(self, audio_file_path: str, session_id: Optional[str] = None) -> dict:
        """
        Non-interactive variant: transcribes a pre-recorded audio file
        (instead of listening live) and runs the agent. Used by the
        /voice API endpoint (api/server.py), where the client uploads an
        audio file rather than this server having mic access.

        Does NOT speak the response aloud (no microphone/speaker access
        assumed on a server) — the caller decides what to do with the
        text response, e.g. return it as JSON, or run TTSEngine.save_to_file
        separately if audio output is wanted back.
        """
        session_id = session_id or str(uuid.uuid4())

        transcribed_text = self.stt.transcribe(audio_file_path)
        if not transcribed_text:
            return {
                "transcribed_text": "",
                "response_text": "Sorry, I couldn't understand the audio. Could you try again?",
                "intent": None,
                "verified": None,
                "session_id": session_id,
            }

        result = self._run_agent(transcribed_text, session_id)

        return {
            "transcribed_text": transcribed_text,
            "response_text": result["final_response"],
            "intent": result.get("intent"),
            "verified": result.get("verified"),
            "session_id": session_id,
        }
