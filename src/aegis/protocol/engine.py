"""Sequence-validation engine.

This is the component that turns "we recognised an action" into
"the experiment is / is not being performed correctly".

Design notes
------------
The engine is deliberately a *pure* state machine: it takes observations and a
timestamp, and returns events. It touches no camera, no audio, no disk. That
makes the safety-critical logic unit-testable without hardware, which is what
lets us claim the sequence validation is verified rather than demoed.

Rules implemented (mapped straight onto the SIH statement):

* ``suggest the next step``      -> ``next_step`` / ``StepArmed`` event
* ``alert when a step is skipped``  -> ``StepSkipped`` (+ ``SafetyBlock`` if the
  skipped step was safety-critical)
* ``out of sequence step is added`` -> ``OutOfSequence`` (repeat of a completed
  step, or an action that belongs to no upcoming step)
* per-step ``timeout``            -> ``StepTimeout`` nudge
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Iterable

from aegis.protocol.spec import Protocol, Step


class StepState(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    SKIPPED = "skipped"
    BLOCKED = "blocked"   # safety-critical step was bypassed; awaiting recovery
    FAILED = "failed"


class Severity(str, Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ProtocolEvent:
    """Anything the engine wants the rest of the system to know about."""

    kind: str
    severity: Severity
    message: str
    speech: str | None = None          # what the voice alert should say
    step_id: int | None = None
    step_name: str | None = None
    confidence: float | None = None
    state_version: int = 0              # monotonically increasing protocol version
    monotonic: float = field(default_factory=time.monotonic)
    detail: dict = field(default_factory=dict)


@dataclass
class Observation:
    """A stabilised recognition result handed to the engine."""

    action: str | None
    confidence: float
    zone: str | None = None
    hand: str | None = None
    source: str = "heuristic"          # heuristic | learned | manual


@dataclass
class StepRecord:
    step: Step
    state: StepState = StepState.PENDING
    started_at: float | None = None    # wall-clock epoch
    finished_at: float | None = None
    confidence: float = 0.0
    source: str = ""
    note: str = ""

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
def _clean_inst(step_or_inst) -> str:
    if not step_or_inst:
        return ""
    text = step_or_inst.instruction if hasattr(step_or_inst, "instruction") else str(step_or_inst)
    text = text.strip().rstrip(".").lower()
    for filler in ("please ", "on the control panel"):
        text = text.replace(filler, "")
    return text.strip()


class ProtocolEngine:
    """Validates that observed actions follow the protocol order."""

    def __init__(self, protocol: Protocol, *, clock=time.monotonic) -> None:
        self.protocol = protocol
        self._clock = clock
        self.records: list[StepRecord] = [StepRecord(step) for step in protocol.steps]
        self.cursor = 0                      # index of the expected step
        self.blocked = False                 # safety block latch
        self.block_reason = ""
        self.completed = False
        self.started_wall: float | None = None
        self.state_version: int = 0          # monotonically increasing protocol version
        self._step_armed_at = self._clock()
        self._dwell_action: str | None = None
        self._dwell_since: float | None = None
        self._last_violation_at = -1e9
        self._last_violation_key = ""
        self._last_reminder_at = self._clock()
        self._timeout_fired = False

    def _make_event(
        self,
        kind: str,
        severity: Severity,
        message: str,
        speech: str | None = None,
        step_id: int | None = None,
        step_name: str | None = None,
        confidence: float | None = None,
        detail: dict | None = None,
    ) -> ProtocolEvent:
        return ProtocolEvent(
            kind=kind,
            severity=severity,
            message=message,
            speech=speech,
            step_id=step_id,
            step_name=step_name,
            confidence=confidence,
            state_version=self.state_version,
            detail=detail or {},
        )

    # ------------------------------------------------------------------ state

    @property
    def current_step(self) -> Step | None:
        if self.completed or self.cursor >= len(self.protocol):
            return None
        return self.protocol[self.cursor]

    @property
    def next_instruction(self) -> str:
        if self.blocked:
            return f"HALTED - {self.block_reason}"
        step = self.current_step
        if step is None:
            return "Experiment complete. All steps verified."
        return step.instruction

    def progress(self) -> tuple[int, int]:
        done = sum(1 for r in self.records if r.state is StepState.DONE)
        return done, len(self.records)

    def snapshot(self) -> dict:
        step = self.current_step
        done, total = self.progress()
        return {
            "experiment": self.protocol.experiment,
            "cursor": self.cursor,
            "state_version": self.state_version,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "completed": self.completed,
            "done": done,
            "total": total,
            "current_step_id": step.id if step else None,
            "current_step_name": step.name if step else None,
            "next_instruction": self.next_instruction,
            "steps": [
                {
                    "id": r.step.id,
                    "name": r.step.name,
                    "state": r.state.value,
                    "confidence": round(r.confidence, 3),
                    "duration_s": None if r.duration_s is None else round(r.duration_s, 2),
                    "safety_critical": r.step.safety_critical,
                    "note": r.note,
                }
                for r in self.records
            ],
        }

    # ----------------------------------------------------------------- helpers

    def _emit_armed(self) -> list[ProtocolEvent]:
        step = self.current_step
        self._step_armed_at = self._clock()
        self._timeout_fired = False
        self._last_reminder_at = self._clock()
        if step is None:
            return []
        self.records[self.cursor].state = StepState.ACTIVE
        self.records[self.cursor].started_at = time.time()
        return [
            self._make_event(
                kind="step_armed",
                severity=Severity.INFO,
                message=f"Next: {step.name} - {step.instruction}",
                speech=None,
                step_id=step.id,
                step_name=step.name,
            )
        ]

    def _rate_limited(self, key: str) -> bool:
        now = self._clock()
        if key == self._last_violation_key and now - self._last_violation_at < self.protocol.out_of_sequence_cooldown_s:
            return True
        self._last_violation_key = key
        self._last_violation_at = now
        return False

    def _reset_dwell(self) -> None:
        self._dwell_action = None
        self._dwell_since = None

    # ------------------------------------------------------------------- API

    def start(self) -> list[ProtocolEvent]:
        self.started_wall = time.time()
        self.state_version += 1
        first_step = self.current_step
        instruction = f" Please {_clean_inst(first_step)}." if first_step else ""
        events = [
            self._make_event(
                kind="session_started",
                severity=Severity.INFO,
                message=f"Session started - {self.protocol.experiment} ({len(self.protocol)} steps)",
                speech=f"Experiment started.{instruction}",
            )
        ]
        events += self._emit_armed()
        return events

    def acknowledge(self) -> list[ProtocolEvent]:
        """Operator clears a safety block after correcting the situation."""
        if not self.blocked:
            return []
        self.state_version += 1
        self.blocked = False
        reason, self.block_reason = self.block_reason, ""
        nxt = self.current_step
        speech = f"Safety block cleared. Please {_clean_inst(nxt)}." if nxt else "Safety block cleared."
        events = [
            self._make_event(
                kind="block_cleared",
                severity=Severity.INFO,
                message=f"Safety block acknowledged by operator ({reason})",
                speech=speech,
                detail={"cleared": reason},
            )
        ]
        events += self._emit_armed()
        return events

    def manual_skip(self) -> list[ProtocolEvent]:
        """Operator deliberately marks the current step as not applicable."""
        step = self.current_step
        if step is None or self.blocked:
            return []
        if not step.allow_manual_skip:
            return [
                self._make_event(
                    kind="skip_refused",
                    severity=Severity.WARNING,
                    message=f"Step {step.id} '{step.name}' is safety-critical and cannot be skipped manually.",
                    speech="This step is safety critical and cannot be skipped.",
                    step_id=step.id,
                    step_name=step.name,
                )
            ]
        self.state_version += 1
        rec = self.records[self.cursor]
        rec.state = StepState.SKIPPED
        rec.finished_at = time.time()
        rec.note = "manual skip"
        rec.source = "manual"
        self.cursor += 1
        self._reset_dwell()
        next_step = self.current_step
        speech = (
            f"Step {step.id} skipped. Please {_clean_inst(next_step)}."
            if next_step
            else f"Step {step.id} skipped. Experiment completed successfully."
        )
        events = [
            self._make_event(
                kind="step_skipped_manual",
                severity=Severity.WARNING,
                message=f"Step {step.id} '{step.name}' skipped by operator.",
                speech=speech,
                step_id=step.id,
                step_name=step.name,
            )
        ]
        events += self._finish_or_arm()
        return events

    def force_complete(self) -> list[ProtocolEvent]:
        """Operator confirms the current step manually (recogniser fallback)."""
        step = self.current_step
        if step is None or self.blocked:
            return []
        return self._complete_current(1.0, "manual", note="operator confirmed")

    def _finish_or_arm(self) -> list[ProtocolEvent]:
        if self.cursor >= len(self.protocol):
            self.completed = True
            return [
                self._make_event(
                    kind="session_completed",
                    severity=Severity.SUCCESS,
                    message="All protocol steps accounted for. Experiment complete.",
                    speech="Experiment completed successfully.",
                )
            ]
        return self._emit_armed()

    def _complete_current(self, confidence: float, source: str, note: str = "") -> list[ProtocolEvent]:
        self.state_version += 1
        step = self.protocol[self.cursor]
        rec = self.records[self.cursor]
        rec.state = StepState.DONE
        rec.finished_at = time.time()
        if rec.started_at is None:
            rec.started_at = rec.finished_at
        rec.confidence = confidence
        rec.source = source
        rec.note = note
        self.cursor += 1
        self._reset_dwell()

        next_step = self.current_step
        if next_step:
            speech_text = f"Step {step.id} completed. Please {_clean_inst(next_step)}."
        else:
            speech_text = f"Step {step.id} completed. Experiment completed successfully."

        events = [
            self._make_event(
                kind="step_completed",
                severity=Severity.SUCCESS,
                message=f"Step {step.id} '{step.name}' verified ({confidence:.0%}, {source}).",
                speech=speech_text,
                step_id=step.id,
                step_name=step.name,
                confidence=confidence,
                detail={"source": source, "outcome": step.outcome_hint or "nominal"},
            )
        ]
        events += self._finish_or_arm()
        return events

    def _handle_forward_jump(self, target: int, confidence: float, source: str) -> list[ProtocolEvent]:
        """An action belonging to a *later* step fired -> steps were skipped."""
        self.state_version += 1
        skipped = list(range(self.cursor, target))
        critical = [i for i in skipped if self.protocol[i].safety_critical]
        names = ", ".join(f"{self.protocol[i].id} ({self.protocol[i].name})" for i in skipped)
        events: list[ProtocolEvent] = []

        if critical:
            first = self.protocol[critical[0]]
            self.blocked = True
            self.block_reason = f"safety-critical step {first.id} '{first.name}' was skipped"
            for i in skipped:
                self.records[i].state = StepState.BLOCKED if i in critical else StepState.SKIPPED
                self.records[i].finished_at = time.time()
                self.records[i].note = (
                    "BYPASSED - safety critical, awaiting recovery" if i in critical else "skipped"
                )
            events.append(
                self._make_event(
                    kind="safety_block",
                    severity=Severity.CRITICAL,
                    message=(
                        f"SAFETY BLOCK. Detected '{self.protocol[target].name}' but "
                        f"safety-critical step {first.id} '{first.name}' was not performed. Halting guidance."
                    ),
                    speech=f"Warning. Step {first.id} has not been completed. Please {_clean_inst(first)} first.",
                    step_id=first.id,
                    step_name=first.name,
                    confidence=confidence,
                    detail={"skipped": names, "trigger": self.protocol[target].name},
                )
            )
            # Cursor stays on the first critical step so the operator is guided back to it.
            self.cursor = critical[0]
            self._reset_dwell()
            return events

        for i in skipped:
            self.records[i].state = StepState.SKIPPED
            self.records[i].finished_at = time.time()
            self.records[i].note = "auto-detected skip"
        events.append(
            self._make_event(
                kind="step_skipped",
                severity=Severity.WARNING,
                message=f"Out of sequence: step(s) {names} were skipped. Continuing from step {self.protocol[target].id}.",
                speech=f"Warning. Step {self.protocol[skipped[0]].id} was skipped. Please {_clean_inst(self.protocol[target])}.",
                step_id=self.protocol[skipped[0]].id,
                step_name=self.protocol[skipped[0]].name,
                detail={"skipped": names},
            )
        )
        self.cursor = target
        events += self._complete_current(confidence, source, note="performed after skip")
        return events

    def observe(self, obs: Observation) -> list[ProtocolEvent]:
        """Feed one stabilised observation. Returns events to publish."""
        now = self._clock()
        events: list[ProtocolEvent] = []

        if self.completed:
            return events

        # --- idle / timeout nudges happen regardless of what was observed ----
        step = self.current_step
        if step is not None and not self.blocked:
            elapsed = now - self._step_armed_at
            reminder_interval = min(7.0, self.protocol.idle_reminder_s)
            if not self._timeout_fired and elapsed > step.timeout_s:
                self._timeout_fired = True
                events.append(
                    self._make_event(
                        kind="step_timeout",
                        severity=Severity.WARNING,
                        message=f"Step {step.id} '{step.name}' has exceeded {step.timeout_s:.0f}s.",
                        speech=f"Step {step.id} is taking longer than expected. Please {step.instruction.lower()}.",
                        step_id=step.id,
                        step_name=step.name,
                    )
                )
            elif now - self._last_reminder_at > reminder_interval:
                self._last_reminder_at = now
                events.append(
                    self._make_event(
                        kind="step_reminder",
                        severity=Severity.INFO,
                        message=f"Reminder - {step.instruction}",
                        speech=f"Reminder. Please {step.instruction.lower()}.",
                        step_id=step.id,
                        step_name=step.name,
                    )
                )

        if obs.action is None:
            # Allow a 0.35s grace window so camera frame flicker does not reset dwell continuity
            if self._dwell_since is not None and (now - self._dwell_since) > 0.35:
                self._reset_dwell()
            return events

        # --- while blocked we only listen for the missed critical step ------
        if self.blocked:
            expected = self.current_step
            if expected is not None and obs.action == expected.action and obs.confidence >= expected.min_confidence:
                self.blocked = False
                reason, self.block_reason = self.block_reason, ""
                events.append(
                    self._make_event(
                        kind="block_cleared",
                        severity=Severity.SUCCESS,
                        message=f"Recovery: missed step {expected.id} '{expected.name}' has now been performed.",
                        speech=f"Step {expected.id} completed.",
                        step_id=expected.id,
                        step_name=expected.name,
                        detail={"cleared": reason},
                    )
                )
                events += self._complete_current(obs.confidence, obs.source, note="recovered after block")
            elif not self._rate_limited(f"blocked:{obs.action}"):
                events.append(
                    self._make_event(
                        kind="blocked_action",
                        severity=Severity.CRITICAL,
                        message=f"Ignored '{obs.action}' - guidance is halted until the missed safety step is completed.",
                        speech=(
                            f"Warning. Step {expected.id} has not been completed. Please {expected.instruction.lower()} first."
                            if expected
                            else "Halted. Complete the missed safety step first."
                        ),
                        confidence=obs.confidence,
                    )
                )
            return events

        # --- unknown action -------------------------------------------------
        candidates = self.protocol.indices_of_action(obs.action)
        if not candidates:
            if not self._rate_limited(f"unknown:{obs.action}"):
                nxt = self.current_step
                speech = (
                    f"Warning. Incorrect sequence. Please {nxt.instruction.lower()} first."
                    if nxt
                    else "Unexpected action detected. This is not part of the protocol."
                )
                events.append(
                    self._make_event(
                        kind="unexpected_action",
                        severity=Severity.WARNING,
                        message=f"Observed '{obs.action}' which is not part of this protocol.",
                        speech=speech,
                        confidence=obs.confidence,
                    )
                )
            self._reset_dwell()
            return events

        # --- expected step? -------------------------------------------------
        expected = self.current_step
        if expected is not None and obs.action == expected.action:
            if obs.confidence < expected.min_confidence and obs.source not in ("demo_hotkey", "manual", "demo"):
                self._reset_dwell()
                return events
            if expected.zone and obs.zone and obs.zone != expected.zone and obs.source not in ("demo_hotkey", "manual", "demo") and obs.confidence < 0.75:
                if not self._rate_limited(f"zone:{obs.action}"):
                    events.append(
                        self._make_event(
                            kind="wrong_zone",
                            severity=Severity.WARNING,
                            message=f"'{expected.name}' seen in zone '{obs.zone}' but expected '{expected.zone}'.",
                            speech=f"Wrong location. Perform this step at the {expected.zone.replace('_', ' ')}.",
                            step_id=expected.id,
                            step_name=expected.name,
                        )
                    )
                self._reset_dwell()
                return events

            # Dwell gate: demo/manual hotkey triggers immediately satisfy dwell
            is_manual_demo = obs.source in ("demo_hotkey", "manual", "demo")
            if not is_manual_demo:
                if self._dwell_action != obs.action:
                    self._dwell_action = obs.action
                    self._dwell_since = now
                if self._dwell_since is not None and (now - self._dwell_since) < expected.dwell_s:
                    return events

            events += self._complete_current(obs.confidence, obs.source)
            return events

        # --- an already-completed step repeated ------------------------------
        past = [i for i in candidates if self.records[i].state in (StepState.DONE, StepState.SKIPPED)
                and i < self.cursor]
        future = [i for i in candidates if i > self.cursor]

        if future and (obs.confidence >= self.protocol[future[0]].min_confidence or obs.source in ("demo_hotkey", "manual", "demo")):
            self._reset_dwell()
            events += self._handle_forward_jump(future[0], obs.confidence, obs.source)
            return events

        if past:
            prev = self.protocol[past[-1]]
            is_manual_demo = obs.source in ("demo_hotkey", "manual", "demo")
            threshold = max(0.80, prev.min_confidence + 0.15) if obs.source == "heuristic" else max(0.68, prev.min_confidence)
            if (obs.confidence >= threshold or is_manual_demo) and not self._rate_limited(f"repeat:{obs.action}"):
                nxt = self.current_step
                speech = (
                    f"Warning. Incorrect sequence. Please {_clean_inst(nxt)} first."
                    if nxt
                    else f"Warning. Incorrect sequence. Step {prev.id} is already completed."
                )
                events.append(
                    self._make_event(
                        kind="out_of_sequence",
                        severity=Severity.WARNING,
                        message=(
                            f"Out of sequence: step {prev.id} '{prev.name}' was repeated. "
                            + (f"Expected step {nxt.id} '{nxt.name}'." if nxt else "")
                        ),
                        speech=speech,
                        step_id=prev.id,
                        step_name=prev.name,
                        confidence=obs.confidence,
                    )
                )
        self._reset_dwell()
        return events
