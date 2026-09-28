"""Hardware-free tests for the sequence-validation logic.

These are the tests that matter: they prove skip detection, out-of-sequence
detection and the safety block work, without needing a camera, a model, or a
GPU. Run them with ``python -m pytest tests -q`` or via RUN_TESTS.bat.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegis.protocol.engine import Observation, ProtocolEngine, StepState
from aegis.protocol.spec import Protocol, Step


def build_protocol() -> Protocol:
    steps = (
        Step(1, "Open box", "open_box", "Open the box.", dwell_s=0.0, min_confidence=0.5),
        Step(2, "Arm interlock", "arm_interlock", "Arm the interlock.", dwell_s=0.0,
             min_confidence=0.5, safety_critical=True, allow_manual_skip=False),
        Step(3, "Insert cell", "insert_cell", "Insert the cell.", dwell_s=0.0, min_confidence=0.5),
        Step(4, "Close box", "close_box", "Close the box.", dwell_s=0.0, min_confidence=0.5),
    )
    return Protocol("Test", "", steps, out_of_sequence_cooldown_s=0.0, idle_reminder_s=1e9)


def kinds(events) -> list[str]:
    return [e.kind for e in events]


def test_happy_path_completes_in_order():
    engine = ProtocolEngine(build_protocol())
    engine.start()
    for action in ("open_box", "arm_interlock", "insert_cell", "close_box"):
        engine.observe(Observation(action, 0.9))
    assert engine.completed
    assert all(r.state is StepState.DONE for r in engine.records)
    done, total = engine.progress()
    assert (done, total) == (4, 4)


def test_dwell_gate_requires_persistence():
    protocol = build_protocol()
    steps = list(protocol.steps)
    steps[0] = Step(1, "Open box", "open_box", "Open the box.", dwell_s=0.5, min_confidence=0.5)
    protocol = Protocol("Test", "", tuple(steps), out_of_sequence_cooldown_s=0.0, idle_reminder_s=1e9)

    now = [100.0]
    engine = ProtocolEngine(protocol, clock=lambda: now[0])
    engine.start()
    engine.observe(Observation("open_box", 0.9))
    assert engine.cursor == 0, "first sighting starts the dwell timer only"
    now[0] += 0.2
    engine.observe(Observation("open_box", 0.9))
    assert engine.cursor == 0, "dwell not yet satisfied"
    now[0] += 0.5
    engine.observe(Observation("open_box", 0.9))
    assert engine.cursor == 1, "dwell satisfied -> step completes"


def test_low_confidence_is_ignored():
    engine = ProtocolEngine(build_protocol())
    engine.start()
    engine.observe(Observation("open_box", 0.2))
    assert engine.cursor == 0


def test_skipping_a_noncritical_step_warns_and_continues():
    protocol = build_protocol()
    steps = list(protocol.steps)
    steps[1] = Step(2, "Label sample", "label_sample", "Label it.", dwell_s=0.0, min_confidence=0.5)
    protocol = Protocol("Test", "", tuple(steps), out_of_sequence_cooldown_s=0.0, idle_reminder_s=1e9)

    engine = ProtocolEngine(protocol)
    engine.start()
    engine.observe(Observation("open_box", 0.9))
    events = engine.observe(Observation("insert_cell", 0.9))     # jumps over step 2
    assert "step_skipped" in kinds(events)
    assert engine.records[1].state is StepState.SKIPPED
    assert engine.records[2].state is StepState.DONE
    assert not engine.blocked
    speech = " ".join(e.speech or "" for e in events)
    assert "skipped" in speech.lower()


def test_skipping_a_safety_critical_step_blocks():
    engine = ProtocolEngine(build_protocol())
    engine.start()
    engine.observe(Observation("open_box", 0.9))
    events = engine.observe(Observation("insert_cell", 0.9))     # skips the interlock
    assert "safety_block" in kinds(events)
    assert engine.blocked
    assert engine.cursor == 1, "cursor rewinds to the missed critical step"
    assert engine.records[1].state is StepState.BLOCKED
    assert any(e.severity.value == "critical" for e in events)


def test_block_is_cleared_by_performing_the_missed_step():
    engine = ProtocolEngine(build_protocol())
    engine.start()
    engine.observe(Observation("open_box", 0.9))
    engine.observe(Observation("insert_cell", 0.9))
    assert engine.blocked

    ignored = engine.observe(Observation("close_box", 0.95))
    assert "blocked_action" in kinds(ignored)
    assert engine.blocked, "other actions must not clear a safety block"

    events = engine.observe(Observation("arm_interlock", 0.9))
    assert "block_cleared" in kinds(events)
    assert not engine.blocked
    assert engine.records[1].state is StepState.DONE


def test_repeating_a_completed_step_is_flagged_out_of_sequence():
    engine = ProtocolEngine(build_protocol())
    engine.start()
    engine.observe(Observation("open_box", 0.9))
    events = engine.observe(Observation("open_box", 0.9))
    assert "out_of_sequence" in kinds(events)
    assert engine.cursor == 1, "cursor must not advance on a repeat"


def test_unknown_action_is_reported_not_ignored():
    engine = ProtocolEngine(build_protocol())
    engine.start()
    events = engine.observe(Observation("juggle_the_payload", 0.95))
    assert "unexpected_action" in kinds(events)
    assert engine.cursor == 0


def test_timeout_nudges_once():
    now = [0.0]
    engine = ProtocolEngine(build_protocol(), clock=lambda: now[0])
    engine.start()
    now[0] = 1000.0
    first = engine.observe(Observation(None, 0.0))
    second = engine.observe(Observation(None, 0.0))
    assert "step_timeout" in kinds(first)
    assert "step_timeout" not in kinds(second), "timeout must not repeat every frame"


def test_manual_skip_refused_on_critical_step():
    engine = ProtocolEngine(build_protocol())
    engine.start()
    engine.observe(Observation("open_box", 0.9))
    events = engine.manual_skip()
    assert "skip_refused" in kinds(events)
    assert engine.cursor == 1


def test_manual_skip_allowed_on_normal_step():
    engine = ProtocolEngine(build_protocol())
    engine.start()
    events = engine.manual_skip()
    assert "step_skipped_manual" in kinds(events)
    assert engine.records[0].state is StepState.SKIPPED
    assert engine.cursor == 1


def test_next_instruction_is_always_available():
    engine = ProtocolEngine(build_protocol())
    events = engine.start()
    assert "step_armed" in kinds(events)
    assert engine.next_instruction == "Open the box."
    engine.observe(Observation("open_box", 0.9))
    assert engine.next_instruction == "Arm the interlock."


def test_snapshot_shape():
    engine = ProtocolEngine(build_protocol())
    engine.start()
    snap = engine.snapshot()
    assert snap["total"] == 4
    assert len(snap["steps"]) == 4
    assert snap["current_step_id"] == 1
    assert snap["steps"][0]["state"] == "active"
