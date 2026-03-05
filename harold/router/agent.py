"""Stateless Haiku router using forced tool_use for structured output.

Classifies spoken transcripts into one of 5 intents:
spawn_session, read_status, list_sessions, kill_session, send_input.
"""

import logging

import anthropic

from harold.config import ROUTER_MAX_RETRIES, ROUTER_MODEL
from harold.router.models import (
    KillSession,
    ListSessions,
    ReadStatus,
    RouterOutput,
    SendInput,
    SpawnSession,
)

logger = logging.getLogger(__name__)

SYSTEM_TEMPLATE = """\
You are Harold's voice command router. Classify spoken transcripts into
exactly one intent and extract structured parameters.

The transcript comes from speech-to-text and may contain recognition errors,
filler words, or imprecise language. Interpret the user's likely intention.

Active sessions:
{session_list}

{project_list}

Intent rules:
1. spawn_session — User wants to start a new coding task. Extract a clean
   task prompt from the transcript. If the user mentions a known project by
   name, set the "project" field to that project name (lowercase). If no
   project is mentioned or the name doesn't match a known project, set
   "project" to null.
2. read_status — User wants a progress/result update on a session. Match
   the session name from the active sessions list. If only one session exists,
   assume that one.
3. list_sessions — User wants to know what sessions are active.
4. kill_session — User wants to stop/cancel a session.
5. send_input — User wants to send a follow-up instruction to a session
   in the "awaiting_input" state. Extract the session name and the
   follow-up message. If only one session is awaiting_input, assume that one.

When matching session names, use fuzzy matching — the user might say
"auth refactor" for "auth-refactor" or just "auth" if unambiguous.

Always call the "route" tool. Never respond with plain text."""

ROUTE_TOOL = {
    "name": "route",
    "description": "Classify the user's voice command into a structured intent.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [
                    "spawn_session",
                    "read_status",
                    "list_sessions",
                    "kill_session",
                    "send_input",
                ],
            },
            "prompt": {
                "type": "string",
                "description": "For spawn_session: the cleaned task description.",
            },
            "name": {
                "type": "string",
                "description": "For read_status, kill_session, send_input: the session name.",
            },
            "message": {
                "type": "string",
                "description": "For send_input: the follow-up instruction.",
            },
            "project": {
                "type": ["string", "null"],
                "description": "For spawn_session: the project name if mentioned.",
            },
        },
        "required": ["intent"],
    },
}

_INTENT_MODELS: dict[str, type[RouterOutput]] = {
    "spawn_session": SpawnSession,
    "read_status": ReadStatus,
    "list_sessions": ListSessions,
    "kill_session": KillSession,
    "send_input": SendInput,
}


class Router:
    """Classifies voice transcripts into structured intents via Haiku."""

    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(max_retries=ROUTER_MAX_RETRIES)

    async def classify(
        self,
        transcript: str,
        session_registry: list[dict[str, str]],
        project_names: list[str] | None = None,
    ) -> RouterOutput | None:
        """Classify a transcript into one of 5 intents.

        Returns None on any failure (API error, no tool_use block, parse failure).
        """
        if not session_registry:
            session_list = "  (none)"
        else:
            session_list = "\n".join(
                f"  - name={s['name']}  state={s['state']}"
                for s in session_registry
            )

        if project_names:
            project_list = "Known projects:\n" + "\n".join(
                f"  - {name}" for name in project_names
            )
        else:
            project_list = ""

        system_prompt = SYSTEM_TEMPLATE.format(
            session_list=session_list,
            project_list=project_list,
        )

        try:
            response = await self._client.messages.create(
                model=ROUTER_MODEL,
                max_tokens=200,
                temperature=0.0,
                system=system_prompt,
                tools=[ROUTE_TOOL],
                tool_choice={"type": "tool", "name": "route"},
                messages=[{"role": "user", "content": transcript}],
            )
        except Exception:
            logger.exception("Router API call failed")
            return None

        # Extract the tool_use block
        for block in response.content:
            if block.type == "tool_use" and block.name == "route":
                return self._parse_tool_input(block.input)

        logger.warning("Router response contained no route tool_use block")
        return None

    def _parse_tool_input(self, raw: dict) -> RouterOutput | None:
        """Parse the raw tool input dict into a typed Pydantic model."""
        intent = raw.get("intent")
        model_cls = _INTENT_MODELS.get(intent)
        if model_cls is None:
            logger.warning("Unknown intent from router: %r", intent)
            return None

        try:
            return model_cls.model_validate(raw)
        except Exception:
            logger.exception("Failed to parse router output: %r", raw)
            return None
