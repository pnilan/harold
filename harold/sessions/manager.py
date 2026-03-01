"""Session manager — routes voice input through a single Claude Agent SDK session.

State machine:
    IDLE ──speak──> spawn session ──> RUNNING
    RUNNING ──speak──> summarize status ──> RUNNING
    RUNNING ──ResultMessage──> ping ──> COMPLETED or ERROR
    COMPLETED ──speak──> summarize result ──> IDLE
    ERROR ──speak──> summarize error ──> IDLE
"""

import asyncio
import enum
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)
from claude_agent_sdk.types import (
    PermissionResultAllow,
    ToolPermissionContext,
    ToolUseBlock,
)

from harold.config import CLAUDE_MAX_BUDGET_USD, DEFAULT_CWD
from harold.sessions.summarizer import Summarizer

logger = logging.getLogger(__name__)


class SessionState(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class Session:
    name: str
    prompt: str
    state: SessionState
    client: ClaudeSDKClient
    message_log: list = field(default_factory=list)
    summary_cursor: int = 0
    result: ResultMessage | None = None
    _task: asyncio.Task | None = None


class SessionManager:
    """Manages a single Claude Agent SDK session driven by voice input."""

    def __init__(
        self,
        on_speak: Callable[[str], Awaitable[None]],
        on_ping_complete: Callable[[], Awaitable[None]],
        on_ping_error: Callable[[], Awaitable[None]],
    ) -> None:
        self._on_speak = on_speak
        self._on_ping_complete = on_ping_complete
        self._on_ping_error = on_ping_error
        self._session: Session | None = None
        self._summarizer = Summarizer()

    @property
    def state(self) -> SessionState:
        if self._session is None:
            return SessionState.IDLE
        return self._session.state

    async def handle_transcript(self, text: str) -> None:
        """Single entry point from the main loop. Routes by current state."""
        state = self.state
        logger.info("handle_transcript state=%s text=%r", state.value, text[:80])

        if state == SessionState.IDLE:
            await self._spawn_session(text)
        elif state == SessionState.RUNNING:
            await self._speak_status()
        elif state == SessionState.COMPLETED:
            await self._speak_final_result()
        elif state == SessionState.ERROR:
            await self._speak_error()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _spawn_session(self, prompt: str) -> None:
        """Create a new SDK client, connect, and start a background task."""
        name = await self._summarizer.generate_session_name(prompt)
        logger.info("Spawning session %r for prompt: %s", name, prompt[:100])

        await self._on_speak(f"Starting session: {name}")

        options = ClaudeAgentOptions(
            cwd=DEFAULT_CWD,
            permission_mode="acceptEdits",
            can_use_tool=self._can_use_tool,
            max_budget_usd=CLAUDE_MAX_BUDGET_USD,
        )
        client = ClaudeSDKClient(options=options)

        session = Session(
            name=name,
            prompt=prompt,
            state=SessionState.RUNNING,
            client=client,
        )
        self._session = session
        session._task = asyncio.create_task(self._run_session(session))

    async def _run_session(self, session: Session) -> None:
        """Background task: query the SDK and collect messages until ResultMessage."""
        try:
            await session.client.connect()
            await session.client.query(session.prompt)

            async for msg in session.client.receive_response():
                session.message_log.append(msg)
                _log_sdk_message(msg)

                if isinstance(msg, ResultMessage):
                    session.result = msg
                    if msg.is_error:
                        session.state = SessionState.ERROR
                        logger.warning("Session %s errored: %s", session.name, msg.subtype)
                        await self._on_ping_error()
                    else:
                        session.state = SessionState.COMPLETED
                        logger.info("Session %s completed: %s", session.name, msg.subtype)
                        await self._on_ping_complete()

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Session %s crashed", session.name)
            session.state = SessionState.ERROR
            await self._on_ping_error()
        finally:
            try:
                await session.client.disconnect()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Summarization callbacks
    # ------------------------------------------------------------------

    async def _speak_status(self) -> None:
        """Summarize new activity since last check and speak it."""
        session = self._session
        if session is None:
            return

        new_messages = session.message_log[session.summary_cursor :]
        summary = await self._summarizer.summarize_progress(session.name, new_messages)
        session.summary_cursor = len(session.message_log)
        await self._on_speak(summary)

    async def _speak_final_result(self) -> None:
        """Summarize the completed session result, speak it, and reset to IDLE."""
        session = self._session
        if session is None:
            return

        summary = await self._summarizer.summarize_result(session.name, session.result)
        await self._on_speak(summary)
        await self._reset()

    async def _speak_error(self) -> None:
        """Summarize the session error, speak it, and reset to IDLE."""
        session = self._session
        if session is None:
            return

        if session.result:
            summary = await self._summarizer.summarize_result(session.name, session.result)
        else:
            summary = f"Session {session.name} encountered an error."
        await self._on_speak(summary)
        await self._reset()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _reset(self) -> None:
        """Clean up the current session and return to IDLE."""
        session = self._session
        if session is None:
            return

        if session._task and not session._task.done():
            session._task.cancel()
            try:
                await session._task
            except asyncio.CancelledError:
                pass

        self._session = None
        logger.info("Session reset — back to IDLE")

    async def shutdown(self) -> None:
        """Interrupt and clean up the active session. Called on Ctrl+C."""
        session = self._session
        if session is None:
            return

        logger.info("Shutting down session %s", session.name)
        # Cancel the task first so it stops iterating, then disconnect.
        # The task's finally block calls disconnect(), but we call it again
        # as a safety net — the SDK guards against double-disconnect.
        if session._task and not session._task.done():
            session._task.cancel()
            try:
                await session._task
            except asyncio.CancelledError:
                pass
        try:
            await session.client.disconnect()
        except Exception:
            pass
        self._session = None

    # ------------------------------------------------------------------
    # Permission callback
    # ------------------------------------------------------------------

    @staticmethod
    async def _can_use_tool(
        tool_name: str, input_data: dict, context: ToolPermissionContext
    ) -> PermissionResultAllow:
        """Allow all tools, log tool use. Hook point for future voice-based approval."""
        logger.info("Tool requested: %s", tool_name)
        return PermissionResultAllow(updated_input=input_data)


def _log_sdk_message(msg) -> None:
    """Log an SDK message at debug level."""
    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                logger.debug("AssistantMessage TextBlock (%d chars)", len(block.text))
            elif isinstance(block, ToolUseBlock):
                logger.debug("AssistantMessage ToolUse: %s", block.name)
    elif isinstance(msg, ResultMessage):
        cost = f"${msg.total_cost_usd:.4f}" if msg.total_cost_usd else "N/A"
        logger.info(
            "ResultMessage subtype=%s cost=%s duration=%dms turns=%d",
            msg.subtype, cost, msg.duration_ms, msg.num_turns,
        )
    else:
        logger.debug("SDK message: %s", type(msg).__name__)
