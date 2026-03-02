"""Pydantic models for router intent classification."""

from typing import Literal

from pydantic import BaseModel


class SpawnSession(BaseModel):
    intent: Literal["spawn_session"]
    prompt: str


class ReadStatus(BaseModel):
    intent: Literal["read_status"]
    name: str


class ListSessions(BaseModel):
    intent: Literal["list_sessions"]


class KillSession(BaseModel):
    intent: Literal["kill_session"]
    name: str


RouterOutput = SpawnSession | ReadStatus | ListSessions | KillSession
