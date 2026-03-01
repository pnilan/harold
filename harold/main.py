import asyncio
import threading

import janus

from harold.audio.listener import AudioListener
from harold.audio.speaker import Speaker


async def main():
    queue: janus.Queue[str] = janus.Queue()
    speaker = Speaker()

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

            # Echo test: speak the transcript back
            # Run in a thread so we don't block the async loop
            await asyncio.to_thread(speaker.speak, transcript)
    except asyncio.CancelledError:
        pass
    finally:
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
