from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


def event_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class RunEvent:
    kind: str
    message: str
    level: str = "info"
    timestamp: str = field(default_factory=event_timestamp)
    trial_id: str | None = None
    phase: str | None = None
    stage_name: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RunObserver(Protocol):
    def emit(self, event: RunEvent) -> None:
        raise NotImplementedError


class NullRunObserver:
    def emit(self, event: RunEvent) -> None:
        del event

