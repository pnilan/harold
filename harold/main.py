import asyncio
import logging

import janus
from dotenv import load_dotenv

load_dotenv()

from harold.audio.listener import AudioListener
from harold.audio.ping import play_complete_ping, play_error_ping
from harold.audio.speaker import Speaker
from harold.config import LOG_LEVEL
from harold.router import Router
from harold.router.models import (
    KillSession,
    ListSessions,
    ReadStatus,
    SpawnSession,
)
from harold.sessions.manager import SessionManager


def _fallback(
    transcript: str, registry: list[dict[str, str]]
) -> SpawnSession | ReadStatus | None:
    """Best-effort fallback when the router API call fails.

    Mirrors Phase 2 behaviour: no sessions → spawn, one running → status.
    """
    if not registry:
        return SpawnSession(intent="spawn_session", prompt=transcript)

    running = [s for s in registry if s["state"] == "running"]
    if len(running) == 1:
        return ReadStatus(intent="read_status", name=running[0]["name"])

    return None


async def main():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    queue: janus.Queue[str] = janus.Queue()
    speaker = Speaker()

    # -- Async wrappers for SessionManager callbacks --

    async def on_speak(text: str) -> None:
        await asyncio.to_thread(speaker.speak, text)

    async def on_ping_complete() -> None:
        await asyncio.to_thread(play_complete_ping)

    async def on_ping_error() -> None:
        await asyncio.to_thread(play_error_ping)

    session_mgr = SessionManager(
        on_speak=on_speak,
        on_ping_complete=on_ping_complete,
        on_ping_error=on_ping_error,
    )
    router = Router()

    def on_transcript(text: str):
        queue.sync_q.put(text)

    def on_recording_start():
        speaker.stop()

    listener = AudioListener(
        on_transcript=on_transcript,
        on_recording_start=on_recording_start,
    )
    listener.start()

    print("Harold is listening. Press Ctrl+C to quit.")

    try:
        while True:
            transcript = await queue.async_q.get()
            print(f"\n> {transcript}")

            # 1. Classify
            registry = session_mgr.get_session_registry()
            action = await router.classify(transcript, registry)

            # 2. Fallback if router fails
            if action is None:
                action = _fallback(transcript, registry)
            if action is None:
                await on_speak(
                    "Sorry, I didn't understand that. Please try again."
                )
                continue

            print(f"  [intent: {action.intent}]")

            # 3. Dispatch
            match action:
                case SpawnSession():
                    await session_mgr.spawn_session(action.prompt)
                case ReadStatus():
                    await session_mgr.read_status(action.name)
                case ListSessions():
                    await session_mgr.list_sessions()
                case KillSession():
                    await session_mgr.kill_session(action.name)

    except asyncio.CancelledError:
        pass
    finally:
        await session_mgr.shutdown()
        listener.stop()
        queue.close()
        await queue.wait_closed()


def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    run()
