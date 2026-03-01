import asyncio
import logging

import janus
from dotenv import load_dotenv

load_dotenv()

from harold.audio.listener import AudioListener
from harold.audio.ping import play_complete_ping, play_error_ping
from harold.audio.speaker import Speaker
from harold.config import LOG_LEVEL
from harold.sessions.manager import SessionManager


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
            print(f"  [state: {session_mgr.state.value}]")

            await session_mgr.handle_transcript(transcript)

            print(f"  [state: {session_mgr.state.value}]")
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
