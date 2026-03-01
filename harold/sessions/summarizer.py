"""Haiku-based summarization for session progress and results.

Uses the Anthropic SDK directly (not the Agent SDK) to call Haiku
for fast, cheap summarization of session activity.
"""

import logging
import re
import time

import anthropic
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
from claude_agent_sdk.types import ToolUseBlock

from harold.config import SUMMARIZER_MODEL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a concise voice assistant summarizer. "
    "Your output will be spoken aloud via text-to-speech. "
    "Never use markdown, bullet points, code blocks, or special formatting. "
    "Use short, natural sentences. Keep responses to 1-3 sentences."
)


class Summarizer:
    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic()

    async def generate_session_name(self, prompt: str) -> str:
        """Generate a 2-3 word kebab-case slug for a session prompt."""
        try:
            response = await self._client.messages.create(
                model=SUMMARIZER_MODEL,
                max_tokens=30,
                system=(
                    "Generate a 2-3 word kebab-case slug describing this task. "
                    "Reply with ONLY the slug, nothing else. "
                    "Example: fix-auth-bug"
                ),
                messages=[{"role": "user", "content": prompt}],
            )
            slug = response.content[0].text.strip().lower()
            # Sanitize: keep only alphanumeric and hyphens
            slug = "".join(c for c in slug if c.isalnum() or c == "-")
            slug = re.sub(r"-{2,}", "-", slug).strip("-")
            if slug:
                return slug
        except Exception as e:
            logger.warning("Failed to generate session name: %s", e)

        return f"session-{int(time.time())}"

    async def summarize_progress(self, session_name: str, messages: list) -> str:
        """Summarize recent session activity for a spoken progress update."""
        digest_parts: list[str] = []
        for msg in messages:
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        digest_parts.append(f"Text: {block.text[:200]}")
                    elif isinstance(block, ToolUseBlock):
                        digest_parts.append(f"Tool used: {block.name}")

        digest = "\n".join(digest_parts)[:2000]
        if not digest:
            return f"Session {session_name} is running but no activity to report yet."

        try:
            response = await self._client.messages.create(
                model=SUMMARIZER_MODEL,
                max_tokens=150,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Summarize what session '{session_name}' is currently doing "
                            f"based on this activity log:\n\n{digest}"
                        ),
                    }
                ],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error("Summarize progress failed: %s", e)
            return f"Session {session_name} is running."

    async def summarize_result(self, session_name: str, result) -> str:
        """Summarize the final result of a completed session."""
        if not isinstance(result, ResultMessage):
            return f"Session {session_name} finished."

        cost = f"${result.total_cost_usd:.3f}" if result.total_cost_usd else "unknown cost"
        duration_s = result.duration_ms / 1000
        result_text = result.result or ""

        detail = (
            f"Session '{session_name}' finished in {duration_s:.1f} seconds, "
            f"costing {cost}, with {result.num_turns} turns. "
        )
        if result_text:
            detail += f"Final output: {result_text[:500]}"

        try:
            response = await self._client.messages.create(
                model=SUMMARIZER_MODEL,
                max_tokens=150,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"Summarize this session result:\n\n{detail}",
                    }
                ],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error("Summarize result failed: %s", e)
            return (
                f"Session {session_name} completed in {duration_s:.1f} seconds "
                f"at {cost}."
            )
