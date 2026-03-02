"""Standalone validation script for the router agent.

Run with: uv run python scripts/validate_router.py

WARNING: Makes real API calls to Haiku. Expected cost ~$0.05 total.
Do not run in CI without a cost-aware gate.

Tests:
  1. Spawn intent — "Fix the auth bug in the login module"
  2. Status intent — "What's the status of auth-refactor?"
  3. List intent — "List my sessions"
  4. Kill intent — "Kill auth-refactor"
  5. Messy STT — "um start a session to uh fix the login"
  6. Single-session ambiguity — "what's happening?" with one session
"""

import asyncio
import time

from dotenv import load_dotenv

load_dotenv()

from harold.router.agent import Router


SAMPLE_REGISTRY = [
    {"name": "auth-refactor", "state": "running"},
    {"name": "fix-tests", "state": "completed"},
]

SINGLE_SESSION_REGISTRY = [
    {"name": "auth-refactor", "state": "running"},
]


async def test_spawn() -> None:
    print("\n=== Test 1: Spawn intent ===")
    router = Router()
    result = await router.classify(
        "Fix the auth bug in the login module",
        session_registry=[],
    )
    assert result is not None, "Router returned None"
    assert result.intent == "spawn_session", f"Expected spawn_session, got {result.intent}"
    print(f"  intent={result.intent} prompt={result.prompt!r}")
    print("PASS")


async def test_status() -> None:
    print("\n=== Test 2: Status intent ===")
    router = Router()
    result = await router.classify(
        "What's the status of auth-refactor?",
        session_registry=SAMPLE_REGISTRY,
    )
    assert result is not None, "Router returned None"
    assert result.intent == "read_status", f"Expected read_status, got {result.intent}"
    assert "auth" in result.name.lower(), f"Expected auth-refactor match, got {result.name!r}"
    print(f"  intent={result.intent} name={result.name!r}")
    print("PASS")


async def test_list() -> None:
    print("\n=== Test 3: List intent ===")
    router = Router()
    result = await router.classify(
        "List my sessions",
        session_registry=SAMPLE_REGISTRY,
    )
    assert result is not None, "Router returned None"
    assert result.intent == "list_sessions", f"Expected list_sessions, got {result.intent}"
    print(f"  intent={result.intent}")
    print("PASS")


async def test_kill() -> None:
    print("\n=== Test 4: Kill intent ===")
    router = Router()
    result = await router.classify(
        "Kill auth-refactor",
        session_registry=SAMPLE_REGISTRY,
    )
    assert result is not None, "Router returned None"
    assert result.intent == "kill_session", f"Expected kill_session, got {result.intent}"
    assert "auth" in result.name.lower(), f"Expected auth-refactor match, got {result.name!r}"
    print(f"  intent={result.intent} name={result.name!r}")
    print("PASS")


async def test_messy_stt() -> None:
    print("\n=== Test 5: Messy STT → spawn ===")
    router = Router()
    result = await router.classify(
        "um start a session to uh fix the login",
        session_registry=[],
    )
    assert result is not None, "Router returned None"
    assert result.intent == "spawn_session", f"Expected spawn_session, got {result.intent}"
    print(f"  intent={result.intent} prompt={result.prompt!r}")
    print("PASS")


async def test_single_session_ambiguity() -> None:
    print("\n=== Test 6: Single-session ambiguity → read_status ===")
    router = Router()
    result = await router.classify(
        "what's happening?",
        session_registry=SINGLE_SESSION_REGISTRY,
    )
    assert result is not None, "Router returned None"
    assert result.intent == "read_status", f"Expected read_status, got {result.intent}"
    print(f"  intent={result.intent} name={result.name!r}")
    print("PASS")


async def run_all() -> None:
    tests = [
        ("spawn", test_spawn),
        ("status", test_status),
        ("list", test_list),
        ("kill", test_kill),
        ("messy_stt", test_messy_stt),
        ("single_session_ambiguity", test_single_session_ambiguity),
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
