"""Protocol specification: the experiment steps the operator must perform.

A protocol is a strictly ordered list of steps. Each step declares:

* the ``action`` label the recogniser must emit for the step to count as done,
* an optional ``zone`` the acting hand must be inside (rack-relative),
* evidence thresholds (confidence, dwell time, timeout),
* whether the step is ``safety_critical`` (skipping it halts the run).

Keeping this in YAML means the protocol can be re-authored for a different
experiment without touching a single line of Python -- which is exactly what
the SIH statement asks for ("a pre-defined experiment").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ProtocolError(ValueError):
    """Raised when a protocol file is structurally invalid."""


@dataclass(frozen=True)
class Step:
    """One verifiable action in the experiment."""

    id: int
    name: str
    action: str
    instruction: str
    zone: str | None = None
    hand: str = "any"  # any | left | right | both
    min_confidence: float = 0.55
    dwell_s: float = 0.45
    timeout_s: float = 60.0
    safety_critical: bool = False
    allow_manual_skip: bool = True
    outcome_hint: str = ""

    @property
    def label(self) -> str:
        return f"{self.id:02d} - {self.name}"


@dataclass(frozen=True)
class Protocol:
    """An ordered, validated experiment protocol."""

    experiment: str
    description: str
    steps: tuple[Step, ...]
    stability_frames: int = 4
    out_of_sequence_cooldown_s: float = 3.0
    idle_reminder_s: float = 25.0
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.steps:
            raise ProtocolError("protocol contains no steps")
        ids = [s.id for s in self.steps]
        if len(set(ids)) != len(ids):
            raise ProtocolError(f"duplicate step ids: {ids}")
        if ids != sorted(ids):
            raise ProtocolError(f"step ids must be ascending, got {ids}")

    @property
    def action_labels(self) -> tuple[str, ...]:
        """Unique action labels, in protocol order. Used as classifier classes."""
        seen: list[str] = []
        for step in self.steps:
            if step.action not in seen:
                seen.append(step.action)
        return tuple(seen)

    @property
    def zone_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.zone for s in self.steps if s.zone))

    def index_of_action(self, action: str) -> int | None:
        """First step index whose action matches ``action`` (None if unknown)."""
        for idx, step in enumerate(self.steps):
            if step.action == action:
                return idx
        return None

    def indices_of_action(self, action: str) -> list[int]:
        return [i for i, s in enumerate(self.steps) if s.action == action]

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int) -> Step:
        return self.steps[index]


def _coerce_step(raw: Any, position: int) -> Step:
    if not isinstance(raw, dict):
        raise ProtocolError(f"step #{position} is not a mapping")
    missing = [k for k in ("id", "name", "action", "instruction") if k not in raw]
    if missing:
        raise ProtocolError(f"step #{position} missing keys: {missing}")
    known = set(Step.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ProtocolError(f"step {raw['id']} has unknown keys: {sorted(unknown)}")
    return Step(
        id=int(raw["id"]),
        name=str(raw["name"]),
        action=str(raw["action"]).strip().lower().replace(" ", "_"),
        instruction=str(raw["instruction"]),
        zone=(str(raw["zone"]) if raw.get("zone") else None),
        hand=str(raw.get("hand", "any")).lower(),
        min_confidence=float(raw.get("min_confidence", 0.55)),
        dwell_s=float(raw.get("dwell_s", 0.45)),
        timeout_s=float(raw.get("timeout_s", 60.0)),
        safety_critical=bool(raw.get("safety_critical", False)),
        allow_manual_skip=bool(raw.get("allow_manual_skip", True)),
        outcome_hint=str(raw.get("outcome_hint", "")),
    )


def load_protocol(path: str | Path) -> Protocol:
    """Read and validate a protocol YAML file."""
    path = Path(path)
    if not path.exists():
        raise ProtocolError(f"protocol file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "steps" not in raw:
        raise ProtocolError(f"{path} has no 'steps' section")
    steps = tuple(_coerce_step(item, i) for i, item in enumerate(raw["steps"]))
    return Protocol(
        experiment=str(raw.get("experiment", path.stem)),
        description=str(raw.get("description", "")),
        steps=steps,
        stability_frames=int(raw.get("stability_frames", 4)),
        out_of_sequence_cooldown_s=float(raw.get("out_of_sequence_cooldown_s", 3.0)),
        idle_reminder_s=float(raw.get("idle_reminder_s", 25.0)),
        source_path=path,
    )
