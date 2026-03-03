"""Session manager — multi-session registry for concurrent Claude Agent SDK sessions.

State machine (per session):
    spawn_session → RUNNING
    RUNNING + ResultMessage(success) → AWAITING_INPUT (keep connected, ping)
    RUNNING + ResultMessage(error) → ERROR (disconnect, ping)
    AWAITING_INPUT + send_input → RUNNING (new query, new bg task)
    AWAITING_INPUT + read_status → speak awaiting summary (keep alive)
    AWAITING_INPUT + kill_session → disconnect, remove
    AWAITING_INPUT + stale timeout → disconnect, remove
    ERROR + read_status → speak error, remove
"""

import asyncio
import enum
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import ClassVar

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)
from claude_agent_sdk.types import (
    Message,
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
    ToolUseBlock,
)

from harold.config import CLAUDE_MAX_BUDGET_USD, DEFAULT_CWD, SESSION_MODEL
from harold.sessions.summarizer import Summarizer

logger = logging.getLogger(__name__)


class SessionState(enum.Enum):
    RUNNING = "running"
    AWAITING_INPUT = "awaiting_input"
    ERROR = "error"


class Session:
    """A single Claude Agent SDK session."""

    def __init__(
        self,
        name: str,
        prompt: str,
        state: SessionState,
        client: ClaudeSDKClient,
    ) -> None:
        self.name = name
        self.prompt = prompt
        self.state = state
        self.client = client
        self.message_log: list[Message] = []
        self.summary_cursor: int = 0
        self.result: ResultMessage | None = None
        self.task: asyncio.Task | None = None
        self.idle_since: float | None = None


class SessionManager:
    """Manages multiple concurrent Claude Agent SDK sessions."""

    _STATUS_COOLDOWN_SECS: ClassVar[float] = 10.0
    _STALE_SESSION_TTL_SECS: ClassVar[float] = 300.0  # 5 minutes

    # Bash commands/patterns that are never allowed without human review.
    _DENIED_BASH_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"\brm\s+-[^\s]*r"),       # rm -r / rm -rf
        re.compile(r"\bgit\s+push\s+--force"), # git push --force
        re.compile(r"\bgit\s+reset\s+--hard"),
        re.compile(r"\bsudo\b"),
        re.compile(r"\bcurl\b.*\|\s*sh"),      # curl | sh
    ]

    def __init__(
        self,
        on_speak: Callable[[str], Awaitable[None]],
        on_ping_complete: Callable[[], Awaitable[None]],
        on_ping_error: Callable[[], Awaitable[None]],
    ) -> None:
        self._on_speak = on_speak
        self._on_ping_complete = on_ping_complete
        self._on_ping_error = on_ping_error
        self._sessions: dict[str, Session] = {}
        self._summarizer = Summarizer()
        self._last_status_times: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def spawn_session(self, prompt: str) -> str:
        """Create a new SDK session and start it in the background.

        Returns the session name.
        """
        await self._reap_stale_sessions()
        name = await self._summarizer.generate_session_name(prompt)
        name = self._deduplicate_name(name)
        logger.info("Spawning session %r for prompt: %s", name, prompt[:100])

        await self._on_speak(f"Starting session: {name}")

        options = ClaudeAgentOptions(
            model=SESSION_MODEL,
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
        self._sessions[name] = session
        session.task = asyncio.create_task(self._run_session(session))
        return name

    async def send_input(self, name: str, message: str) -> None:
        """Send a follow-up message to a session in AWAITING_INPUT state."""
        session = self._resolve_session(name)
        if session is None:
            await self._on_speak(f"No session found matching {name}")
            return

        if session.state != SessionState.AWAITING_INPUT:
            await self._on_speak(
                f"Session {session.name} is not awaiting input. "
                f"It is currently {session.state.value}."
            )
            return

        # Defensive: cancel lingering task if somehow still alive
        if session.task and not session.task.done():
            logger.warning(
                "Session %s has a lingering task despite AWAITING_INPUT; cancelling",
                session.name,
            )
            session.task.cancel()
            try:
                await session.task
            except asyncio.CancelledError:
                pass

        session.state = SessionState.RUNNING
        session.idle_since = None
        session.summary_cursor = len(session.message_log)
        logger.info("Sending follow-up to session %s: %s", session.name, message[:100])
        await self._on_speak(f"Sending input to {session.name}")

        session.task = asyncio.create_task(
            self._run_followup(session, message)
        )

    async def read_status(self, name: str) -> None:
        """Speak the status of a session."""
        session = self._resolve_session(name)
        if session is None:
            await self._on_speak(f"No session found matching {name}")
            return

        if session.state == SessionState.RUNNING:
            await self._speak_progress(session)
        elif session.state == SessionState.AWAITING_INPUT:
            summary = await self._summarizer.summarize_awaiting(
                session.name, session.result
            )
            await self._on_speak(summary)
            # Do NOT remove — session stays alive for potential follow-up
        elif session.state == SessionState.ERROR:
            if session.result:
                summary = await self._summarizer.summarize_result(
                    session.name, session.result
                )
            else:
                summary = f"Session {session.name} encountered an error."
            await self._on_speak(summary)
            self._remove_session(session.name)

    async def list_sessions(self) -> None:
        """Speak the name and state of all sessions."""
        if not self._sessions:
            await self._on_speak("No active sessions.")
            return

        state_labels = {
            SessionState.RUNNING: "running",
            SessionState.AWAITING_INPUT: "awaiting input",
            SessionState.ERROR: "errored",
        }
        parts = [
            f"{s.name} is {state_labels.get(s.state, s.state.value)}"
            for s in self._sessions.values()
        ]
        await self._on_speak(". ".join(parts) + ".")

    async def kill_session(self, name: str) -> None:
        """Cancel and remove a session."""
        session = self._resolve_session(name)
        if session is None:
            await self._on_speak(f"No session found matching {name}")
            return

        await self._cancel_session(session)
        self._remove_session(session.name)
        await self._on_speak(f"Session {session.name} killed.")

    def get_session_registry(self) -> list[dict[str, str]]:
        """Return session names and states for the router system prompt."""
        return [
            {"name": s.name, "state": s.state.value}
            for s in self._sessions.values()
        ]

    async def shutdown(self) -> None:
        """Cancel and disconnect ALL sessions. Called on Ctrl+C."""
        for session in list(self._sessions.values()):
            logger.info("Shutting down session %s", session.name)
            await self._cancel_session(session)
        self._sessions.clear()

    # ------------------------------------------------------------------
    # Session lifecycle (background tasks)
    # ------------------------------------------------------------------

    async def _run_session(self, session: Session) -> None:
        """Background task: connect, query, and consume the first turn."""
        try:
            await session.client.connect()
            await session.client.query(session.prompt)
            await self._consume_turn(session)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Session %s crashed", session.name)
            session.state = SessionState.ERROR
            session.idle_since = time.monotonic()
            await self._on_ping_error()
            await self._disconnect_quiet(session)

    async def _run_followup(self, session: Session, message: str) -> None:
        """Background task: send a follow-up query and consume the turn."""
        try:
            await session.client.query(message)
            await self._consume_turn(session)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Session %s follow-up crashed", session.name)
            session.state = SessionState.ERROR
            session.idle_since = time.monotonic()
            await self._on_ping_error()
            await self._disconnect_quiet(session)

    async def _consume_turn(self, session: Session) -> None:
        """Consume SDK messages until ResultMessage. Transitions state accordingly."""
        async for msg in session.client.receive_response():
            session.message_log.append(msg)
            _log_sdk_message(msg)

            if isinstance(msg, ResultMessage):
                session.result = msg
                if msg.is_error:
                    session.state = SessionState.ERROR
                    session.idle_since = time.monotonic()
                    logger.warning("Session %s errored: %s", session.name, msg.subtype)
                    await self._on_ping_error()
                    await self._disconnect_quiet(session)
                else:
                    session.state = SessionState.AWAITING_INPUT
                    session.idle_since = time.monotonic()
                    logger.info("Session %s turn complete, awaiting input: %s", session.name, msg.subtype)
                    await self._on_ping_complete()
                return

        # SDK iterator ended without yielding a ResultMessage
        if session.state == SessionState.RUNNING:
            logger.warning("Session %s: receive_response ended without ResultMessage", session.name)

    # ------------------------------------------------------------------
    # Status summarization
    # ------------------------------------------------------------------

    async def _speak_progress(self, session: Session) -> None:
        """Summarize new activity since last check, with cooldown."""
        now = time.monotonic()
        last = self._last_status_times.get(session.name, 0.0)
        if now - last < self._STATUS_COOLDOWN_SECS:
            await self._on_speak("Still working, please wait.")
            return
        self._last_status_times[session.name] = now

        new_messages = session.message_log[session.summary_cursor:]
        summary = await self._summarizer.summarize_progress(
            session.name, new_messages
        )
        session.summary_cursor = len(session.message_log)
        await self._on_speak(summary)

    # ------------------------------------------------------------------
    # Name resolution
    # ------------------------------------------------------------------

    def _resolve_session(self, name: str) -> Session | None:
        """Fuzzy-match a session name.

        Priority: exact → case-insensitive → unique substring.
        """
        # Exact match
        if name in self._sessions:
            return self._sessions[name]

        # Case-insensitive
        lower = name.lower()
        for key, session in self._sessions.items():
            if key.lower() == lower:
                return session

        # Substring (only if exactly one match)
        matches = [
            s for key, s in self._sessions.items()
            if lower in key.lower()
        ]
        if len(matches) == 1:
            return matches[0]

        return None

    def _deduplicate_name(self, name: str) -> str:
        """Append -2, -3, etc. if the name already exists."""
        if name not in self._sessions:
            return name
        counter = 2
        while f"{name}-{counter}" in self._sessions:
            counter += 1
        return f"{name}-{counter}"

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------

    def _remove_session(self, name: str) -> None:
        """Remove a session from the registry and clean up status timers."""
        self._sessions.pop(name, None)
        self._last_status_times.pop(name, None)
        logger.info("Session %s removed from registry", name)

    async def _reap_stale_sessions(self) -> None:
        """Remove sessions that have been idle for too long.

        AWAITING_INPUT sessions still have a live SDK connection and must
        be disconnected before removal.
        """
        now = time.monotonic()
        stale = [
            name
            for name, s in self._sessions.items()
            if s.state in (SessionState.AWAITING_INPUT, SessionState.ERROR)
            and s.idle_since is not None
            and now - s.idle_since > self._STALE_SESSION_TTL_SECS
        ]
        for name in stale:
            session = self._sessions[name]
            logger.info("Reaping stale session %s (state=%s)", name, session.state.value)
            if session.state == SessionState.AWAITING_INPUT:
                await self._disconnect_quiet(session)
            self._remove_session(name)

    async def _disconnect_quiet(self, session: Session) -> None:
        """Disconnect a session's client, suppressing errors."""
        try:
            await session.client.disconnect()
        except Exception:
            pass

    async def _cancel_session(self, session: Session) -> None:
        """Cancel a session's background task and disconnect."""
        if session.task and not session.task.done():
            session.task.cancel()
            try:
                await session.task
            except asyncio.CancelledError:
                pass
        await self._disconnect_quiet(session)

    # ------------------------------------------------------------------
    # Permission callback
    # ------------------------------------------------------------------

    @classmethod
    async def _can_use_tool(
        cls, tool_name: str, input_data: dict, context: ToolPermissionContext
    ) -> PermissionResult:
        """Check tool requests against a denylist, log usage.

        Blocks destructive bash patterns (rm -rf, git push --force, sudo, etc.).
        Hook point for future voice-based approval flow.
        """
        logger.info("Tool requested: %s | input: %s", tool_name, input_data)

        if tool_name == "Bash":
            command = input_data.get("command", "")
            for pattern in cls._DENIED_BASH_PATTERNS:
                if pattern.search(command):
                    logger.warning("Denied tool %s — matched pattern %r: %s", tool_name, pattern.pattern, command)
                    return PermissionResultDeny(message=f"Command blocked by safety denylist: {pattern.pattern}")

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
