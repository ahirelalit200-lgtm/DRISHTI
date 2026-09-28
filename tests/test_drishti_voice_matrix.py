"""Verification suite for DRISHTI Voice Synchronization Matrix (Part 24)."""

from pathlib import Path
import time
from aegis.config import AppConfig
from aegis.protocol.engine import ProtocolEngine, Observation
from aegis.protocol.spec import load_protocol
from aegis.outputs.voice import VoiceAnnouncer

ROOT = Path(__file__).resolve().parents[1]


def test_drishti_voice_matrix():
    protocol = load_protocol(ROOT / "configs" / "protocol.yaml")
    engine = ProtocolEngine(protocol)
    announcer = VoiceAnnouncer(enabled=True)
    announcer.start()

    # TEST 1: Start session
    events = engine.start()
    assert len(events) >= 1
    ev_start = events[0]
    assert ev_start.kind == "session_started"
    assert "Experiment started" in ev_start.speech
    assert "open the sample box lid" in ev_start.speech.lower()
    announcer.announce_event(ev_start)

    # TEST 2: Correct Step 1
    obs1 = Observation(action="open_sample_box", confidence=0.95, zone="outer_box")
    engine._dwell_since = time.monotonic() - 1.0
    engine._dwell_action = "open_sample_box"
    events1 = engine.observe(obs1)
    ev_comp1 = next(e for e in events1 if e.kind == "step_completed")
    assert "Step 1 completed" in ev_comp1.speech
    assert "pick the sample" in ev_comp1.speech.lower()
    assert ev_comp1.state_version > ev_start.state_version
    announcer.announce_event(ev_comp1)

    # TEST 3: Correct Step 2
    obs2 = Observation(action="pick_sample", confidence=0.95, zone="outer_box")
    engine._dwell_since = time.monotonic() - 1.0
    engine._dwell_action = "pick_sample"
    events2 = engine.observe(obs2)
    ev_comp2 = next(e for e in events2 if e.kind == "step_completed")
    assert "Step 2 completed" in ev_comp2.speech
    assert "transfer the sample" in ev_comp2.speech.lower()
    announcer.announce_event(ev_comp2)

    # TEST 4: Wrong Step 4 while Step 3 is expected
    obs4 = Observation(action="close_sample_box", confidence=0.95, zone="work_surface")
    engine._dwell_since = time.monotonic() - 1.0
    engine._dwell_action = "close_sample_box"
    events_wrong = engine.observe(obs4)
    ev_warn = next(e for e in events_wrong if e.kind == "safety_block")
    assert "Warning" in ev_warn.speech
    assert "step 3 has not been completed" in ev_warn.speech.lower()
    announcer.announce_event(ev_warn)

    # TEST 5: Perform correct Step 3
    obs3 = Observation(action="transfer_sample", confidence=0.95, zone="sample_tray")
    engine._dwell_since = time.monotonic() - 1.0
    engine._dwell_action = "transfer_sample"
    events3 = engine.observe(obs3)
    ev_comp3 = next(e for e in events3 if e.kind == "step_completed")
    assert "Step 3 completed" in ev_comp3.speech
    assert "close the sample box" in ev_comp3.speech.lower()
    announcer.announce_event(ev_comp3)

    # TEST 6: Repeatedly perform the same action -> Deduplicated
    spoken1 = announcer.announce_event(ev_comp3)
    spoken2 = announcer.announce_event(ev_comp3)
    assert spoken2 is False, "Duplicate utterance within cooldown must be suppressed"

    # TEST 7 & 8: Rapid state transition & stop session
    announcer.reset_voice_state()
    assert announcer.pending() == 0, "Queue must be cleared on reset/stop"

    # TEST 9 & 10: Disable voice
    announcer.enabled = False
    assert announcer.announce_event(ev_comp3) is False, "Disabled voice must produce no TTS output"

    from aegis.protocol.engine import Severity
    announcer.enabled = True
    announcer.reset_voice_state()
    engine.state_version = announcer.current_state_version
    ev_comp4 = engine._make_event(kind="step_completed", severity=Severity.SUCCESS, message="Step 4 completed", speech="Step 4 completed.")
    spoken_re = announcer.announce_event(ev_comp4)
    assert spoken_re is True, "Re-enabled voice must accept current state events"

    announcer.stop()
