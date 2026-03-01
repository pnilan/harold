"""Standalone validation script for the Claude Agent SDK.

Run with: uv run python scripts/validate_sdk.py

WARNING: Makes real API calls. Expected cost ~$0.30 total.
Do not run in CI without a cost-aware gate.

Tests:
  1. Connect + single query
  2. Multi-turn session persistence
  3. Permission callback fires
  4. Interrupt + clean termination
"""

import asyncio
import time

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)
from claude_agent_sdk.types import (
    PermissionResultAllow,
    SystemMessage,
    ToolPermissionContext,
    ToolUseBlock,
)


def log_message(msg) -> None:
    """Log a message in the standard format."""
    if isinstance(msg, SystemMessage):
        sid = getattr(msg, "session_id", "?")
        subtype = getattr(msg, "subtype", "?")
        print(f"  [SystemMessage] subtype={subtype}, session_id={sid}")
    elif isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                preview = block.text[:80].replace("\n", " ")
                print(f"  [AssistantMessage] TextBlock: \"{preview}\" ({len(block.text)} chars)")
            elif isinstance(block, ToolUseBlock):
                print(f"  [AssistantMessage] ToolUse: {block.name}")
            else:
                print(f"  [AssistantMessage] {type(block).__name__}")
    elif isinstance(msg, ResultMessage):
        cost = f"${msg.total_cost_usd:.4f}" if msg.total_cost_usd else "N/A"
        print(
            f"  [ResultMessage] subtype={msg.subtype}, cost={cost}, "
            f"duration={msg.duration_ms}ms, turns={msg.num_turns}"
        )
    else:
        print(f"  [{type(msg).__name__}]")


# ---------------------------------------------------------------------------
# Test 1: Connect + single query
# ---------------------------------------------------------------------------
async def test_single_query() -> None:
    print("\n=== Test 1: Connect + single query ===")
    options = ClaudeAgentOptions(max_turns=1, max_budget_usd=0.05)

    async with ClaudeSDKClient(options=options) as client:
        await client.query("What is 2+2? Reply with just the number.")

        result = None
        async for msg in client.receive_response():
            log_message(msg)
            if isinstance(msg, ResultMessage):
                result = msg

    assert result is not None, "No ResultMessage received"
    assert result.subtype == "success", f"Expected success, got {result.subtype}"
    print("PASS")


# ---------------------------------------------------------------------------
# Test 2: Multi-turn session persistence
# ---------------------------------------------------------------------------
async def test_multi_turn() -> None:
    print("\n=== Test 2: Multi-turn session persistence ===")
    secret = "pineapple"
    options = ClaudeAgentOptions(max_turns=2, max_budget_usd=0.10)

    async with ClaudeSDKClient(options=options) as client:
        # Turn 1: ask Claude to remember a word
        await client.query(
            f"Remember this secret word: {secret}. "
            "Just reply 'Got it' and nothing else."
        )
        result1 = None
        async for msg in client.receive_response():
            log_message(msg)
            if isinstance(msg, ResultMessage):
                result1 = msg
        assert result1 is not None and result1.subtype == "success"

        # Turn 2: ask what the word was
        await client.query("What was the secret word I told you? Reply with just the word.")
        answer_text = ""
        result2 = None
        async for msg in client.receive_response():
            log_message(msg)
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        answer_text += block.text
            if isinstance(msg, ResultMessage):
                result2 = msg

        assert result2 is not None and result2.subtype == "success"
        assert secret in answer_text.lower(), (
            f"Expected '{secret}' in response, got: {answer_text}"
        )
    print("PASS")


# ---------------------------------------------------------------------------
# Test 3: Permission callback fires
# ---------------------------------------------------------------------------
async def test_permission_callback() -> None:
    print("\n=== Test 3: Permission callback ===")
    tools_seen: list[str] = []

    async def on_tool(
        tool_name: str, input_data: dict, context: ToolPermissionContext
    ) -> PermissionResultAllow:
        tools_seen.append(tool_name)
        print(f"  [PermissionCallback] tool={tool_name}")
        return PermissionResultAllow(updated_input=input_data)

    options = ClaudeAgentOptions(
        max_turns=3,
        max_budget_usd=0.10,
        can_use_tool=on_tool,
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query(
            "List the files in the current directory using the Bash tool. "
            "Just run `ls` and report back."
        )
        result = None
        async for msg in client.receive_response():
            log_message(msg)
            if isinstance(msg, ResultMessage):
                result = msg

    assert result is not None, "No ResultMessage received"
    assert len(tools_seen) > 0, "Permission callback never fired"
    print(f"  Tools seen: {tools_seen}")
    print("PASS")


# ---------------------------------------------------------------------------
# Test 4: Interrupt + clean termination
# ---------------------------------------------------------------------------
async def test_interrupt() -> None:
    print("\n=== Test 4: Interrupt ===")
    options = ClaudeAgentOptions(max_turns=5, max_budget_usd=0.10)
    message_count = 0
    result = None

    async with ClaudeSDKClient(options=options) as client:
        await client.query(
            "Write a very long essay about the history of mathematics. "
            "Make it at least 2000 words."
        )

        interrupted = False
        async for msg in client.receive_response():
            message_count += 1
            log_message(msg)
            if isinstance(msg, ResultMessage):
                result = msg
            elif not interrupted and message_count >= 2:
                print("  >> Sending interrupt...")
                await client.interrupt()
                interrupted = True

    assert result is not None, "No ResultMessage received after interrupt"
    print(f"  Got ResultMessage after interrupt (subtype={result.subtype})")
    print("PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run_all() -> None:
    tests = [
        ("single_query", test_single_query),
        ("multi_turn", test_multi_turn),
        ("permission_callback", test_permission_callback),
        ("interrupt", test_interrupt),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            t0 = time.monotonic()
            await fn()
            elapsed = time.monotonic() - t0
            print(f"  ({elapsed:.1f}s)")
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(run_all())
