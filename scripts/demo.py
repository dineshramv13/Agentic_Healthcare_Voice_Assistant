"""
scripts/demo.py

Interactive CLI demo loop. Two modes:
    python scripts/demo.py --mode text    (default — type messages, see responses)
    python scripts/demo.py --mode voice    (speak into your mic, hear AI reply)

This is the live, ready demo: text mode needs only what Phases 1-3
installed; voice mode additionally needs Phase 4's voice dependencies
(openai-whisper, pyttsx3, sounddevice, ffmpeg).

Usage:
    python scripts/demo.py
    python scripts/demo.py --mode voice
    python scripts/demo.py --mode voice --session-id my-test-session
"""

import sys
import os
import argparse
import logging
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def run_text_demo(session_id: str):
    from agent.graph import get_graph
    from memory.session import SessionMemory

    print("=" * 60)
    print("AI-Local — Text Demo Mode")
    print("Type your message and press Enter. Type 'quit' to exit.")
    print(f"Session ID: {session_id}")
    print("=" * 60)

    graph = get_graph()
    memory = SessionMemory()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        trace_id = str(uuid.uuid4())
        history_text = memory.get_history_as_text(session_id, last_n=6)

        result = graph.invoke({
            "user_message": user_input,
            "session_id": session_id,
            "trace_id": trace_id,
            "conversation_history": history_text,
            "retry_count": 0,
        })

        final_response = result.get("final_response") or result.get("response") or "(no response generated)"
        memory.add_turn(session_id, role="user", content=user_input, intent=result.get("intent"))
        memory.add_turn(session_id, role="assistant", content=final_response)

        print(f"\n[intent: {result.get('intent')} | verified: {result.get('verified')} | retries: {result.get('retry_count', 0)}]")
        print(f"AI: {final_response}")


def run_voice_demo(session_id: str):
    from voice.pipeline import VoicePipeline

    print("=" * 60)
    print("AI-Local — Voice Demo Mode")
    print("Speak after 'Listening...' appears. Press Ctrl+C to exit.")
    print(f"Session ID: {session_id}")
    print("=" * 60)
    print("\nLoading Whisper model and TTS engine (first run may take a moment)...")

    pipeline = VoicePipeline()

    print("Ready. Try saying things like:")
    print('  - "I need to book an appointment"')
    print('  - "How do I get a repeat prescription?"')
    print("  - \"I think I'm having a heart attack\" (tests emergency routing — instant, no LLM)")

    while True:
        try:
            result = pipeline.listen_and_respond(session_id=session_id)
            print(f"[intent: {result.get('intent')} | verified: {result.get('verified')}]")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError during voice turn: {e}")
            print("Continuing to listen... (Ctrl+C to exit)")


def main():
    parser = argparse.ArgumentParser(description="AI-Local interactive demo")
    parser.add_argument("--mode", choices=["text", "voice"], default="text", help="Demo mode (default: text)")
    parser.add_argument("--session-id", default=None, help="Reuse a specific session ID (default: random new session)")
    args = parser.parse_args()

    session_id = args.session_id or f"demo-{uuid.uuid4().hex[:8]}"

    if args.mode == "text":
        run_text_demo(session_id)
    else:
        run_voice_demo(session_id)


if __name__ == "__main__":
    main()
