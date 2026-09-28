"""Feature extraction, recogniser fusion, and logbook artefact tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegis.outputs.logbook import Logbook
from aegis.perception.features import FEATURE_VERSION, FeatureExtractor, WindowBuffer
from aegis.perception.landmarks import HandObservation, PerceptionResult, PoseObservation
from aegis.perception.rack_frame import identity_frame
from aegis.perception.recognizer import (
    HeuristicRecognizer,
    Recognition,
    RecognitionEngine,
    StabilityFilter,
)
from aegis.perception.zones import Zone, ZoneSet
from aegis.protocol.engine import ProtocolEngine, Severity
from aegis.protocol.spec import Protocol, Step


def make_zones() -> ZoneSet:
    return ZoneSet([
        Zone("bay_a", np.array([[0.0, 0.0], [0.45, 0.0], [0.45, 0.5], [0.0, 0.5]], np.float32)),
        Zone("bay_b", np.array([[0.55, 0.5], [1.0, 0.5], [1.0, 1.0], [0.55, 1.0]], np.float32)),
    ])


def synth_hand(label: str, centre_rack, aperture: float = 1.0) -> HandObservation:
    """A 21-point hand whose grip aperture is controllable."""
    cx, cy = centre_rack
    pts = np.zeros((21, 2), np.float32)
    pts[0] = [cx, cy]                       # wrist
    pts[5] = [cx + 0.05, cy - 0.02]         # index MCP  -> defines span
    pts[17] = [cx - 0.05, cy - 0.02]        # pinky MCP
    pts[9] = [cx, cy - 0.06]                # middle MCP -> palm axis
    reach = 0.10 * aperture
    for i, tip in enumerate((4, 8, 12, 16, 20)):
        angle = -0.9 + i * 0.45
        pts[tip] = [cx + reach * np.cos(angle), cy + reach * np.sin(angle)]
    for i in range(21):
        if not pts[i].any():
            pts[i] = [cx, cy]
    hand = HandObservation(label, 0.9, pts.copy())
    hand.points_rack = pts.copy()
    return hand


def synth_result(hands, frame=0) -> PerceptionResult:
    return PerceptionResult(frame_index=frame, timestamp=float(frame), hands=list(hands))


# ------------------------------------------------------------------ features

def test_feature_width_matches_declaration():
    extractor = FeatureExtractor(make_zones())
    vec = extractor.extract(synth_result([synth_hand("right", (0.2, 0.25))]))
    assert vec.shape == (extractor.dimension,)
    assert len(extractor.feature_names()) == extractor.dimension


def test_features_are_finite_even_with_no_detections():
    extractor = FeatureExtractor(make_zones())
    vec = extractor.extract(synth_result([]))
    assert np.isfinite(vec).all()
    assert vec.shape == (extractor.dimension,)


def test_zone_occupancy_appears_in_the_feature_vector():
    extractor = FeatureExtractor(make_zones())
    names = extractor.feature_names()
    idx_a = names.index("right_in_bay_a")
    idx_b = names.index("right_in_bay_b")

    in_a = extractor.extract(synth_result([synth_hand("right", (0.2, 0.25))]))
    assert in_a[idx_a] > in_a[idx_b]

    extractor.reset()
    in_b = extractor.extract(synth_result([synth_hand("right", (0.8, 0.75))]))
    assert in_b[idx_b] > in_b[idx_a]


def test_grip_aperture_responds_to_closing_the_hand():
    open_hand = synth_hand("right", (0.5, 0.5), aperture=1.4)
    shut_hand = synth_hand("right", (0.5, 0.5), aperture=0.3)
    assert open_hand.grip_aperture() > shut_hand.grip_aperture()


def test_window_buffer_left_pads_until_full():
    buf = WindowBuffer(6, 3)
    assert not buf.filled
    buf.push(np.array([1.0, 2.0, 3.0], np.float32))
    arr = buf.array()
    assert arr.shape == (6, 3)
    assert np.allclose(arr[0], arr[-1]), "short buffers pad with the earliest frame"
    for _ in range(6):
        buf.push(np.zeros(3, np.float32))
    assert buf.filled


def test_feature_version_is_recorded_for_compatibility_checks():
    assert isinstance(FEATURE_VERSION, int) and FEATURE_VERSION >= 1


# --------------------------------------------------------------- recognisers

def protocol_two_zones() -> Protocol:
    return Protocol(
        "Zoned",
        "",
        (
            Step(1, "Grasp at A", "grasp_part", "Grasp at bay A.", zone="bay_a", dwell_s=0.0),
            Step(2, "Insert at B", "insert_part", "Insert at bay B.", zone="bay_b", dwell_s=0.0),
        ),
    )


def test_heuristic_prefers_the_step_whose_zone_the_hand_occupies():
    zones = make_zones()
    protocol = protocol_two_zones()
    recogniser = HeuristicRecognizer(protocol, zones)

    for _ in range(10):
        recogniser.update_history(synth_result([synth_hand("right", (0.2, 0.25))]))
    score_a, _, _ = recogniser.score_step(protocol[0], synth_result([synth_hand("right", (0.2, 0.25))]))
    score_b, _, _ = recogniser.score_step(protocol[1], synth_result([synth_hand("right", (0.2, 0.25))]))
    assert score_a > score_b, "hand in bay_a should favour the bay_a step"


def test_heuristic_returns_nothing_without_hands():
    recogniser = HeuristicRecognizer(protocol_two_zones(), make_zones())
    out = recogniser.recognise(synth_result([]), None, list(protocol_two_zones().steps))
    assert out.action is None
    assert out.confidence == 0.0


def test_stability_filter_suppresses_single_frame_spikes():
    filt = StabilityFilter(frames=3)
    assert filt.push(Recognition("grasp_part", 0.9, "heuristic")).action is None
    assert filt.push(Recognition("grasp_part", 0.9, "heuristic")).action is None
    assert filt.push(Recognition("grasp_part", 0.9, "heuristic")).action == "grasp_part"

    # one disagreeing frame breaks the streak
    assert filt.push(Recognition("insert_part", 0.9, "heuristic")).action is None


def test_stability_filter_averages_confidence_over_the_streak():
    filt = StabilityFilter(frames=2)
    filt.push(Recognition("grasp_part", 0.6, "heuristic"))
    out = filt.push(Recognition("grasp_part", 0.8, "heuristic"))
    assert out.action == "grasp_part"
    assert out.confidence == pytest.approx(0.7, abs=1e-6)


def test_engine_falls_back_to_heuristic_without_a_learned_model():
    engine = RecognitionEngine(protocol_two_zones(), make_zones(), stability_frames=1)
    assert "Tier 0" in engine.tier
    out = engine.process(synth_result([synth_hand("right", (0.2, 0.25))]), cursor=0)
    assert out.source in ("heuristic", "none")


def test_engine_reports_why_a_model_is_unusable():
    from aegis.perception.recognizer import LearnedRecognizer

    learned = LearnedRecognizer(Path("does/not/exist.onnx"), Path("nope.json"))
    engine = RecognitionEngine(protocol_two_zones(), make_zones(), learned=learned)
    assert "Tier 0" in engine.tier
    assert engine.tier_detail(), "must explain why the model is not loaded"


def test_protocol_window_covers_neighbouring_steps():
    steps = tuple(
        Step(i + 1, f"S{i+1}", f"a{i+1}", "do it.", dwell_s=0.0) for i in range(6)
    )
    engine = RecognitionEngine(Protocol("P", "", steps), make_zones(), neighbourhood=2)
    window = engine.protocol_window(cursor=3)
    ids = [s.id for s in window]
    assert ids == [2, 3, 4, 5, 6], "window should span cursor +/- neighbourhood"


# ------------------------------------------------------------------ logbook

def test_logbook_writes_all_three_artifacts():
    protocol = protocol_two_zones()
    with tempfile.TemporaryDirectory() as tmp:
        book = Logbook(Path(tmp), "sess_test", protocol.experiment, len(protocol))
        engine = ProtocolEngine(protocol)
        for event in engine.start():
            book.write_event(event)
        book.write_note("camera", "test note")
        report = book.finalise(engine.snapshot(), {"Frames": 42})

        assert report.exists()
        text = report.read_text(encoding="utf-8")
        assert "EXPERIMENT SESSION REPORT" in text
        assert "Frames" in text

        log_text = book.text_path.read_text(encoding="utf-8")
        assert "UTC_TIMESTAMP" in log_text, "header row must be present"
        assert "sess_test" in log_text

        lines = [json.loads(l) for l in book.jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert lines[0]["type"] == "session_header"
        assert any(l["type"] == "session_summary" for l in lines)
        assert all("type" in l for l in lines), "every JSONL line must be a typed object"


def test_report_verdict_reflects_a_bypassed_critical_step():
    protocol = Protocol(
        "Crit", "",
        (
            Step(1, "Prep", "prep", "Prep.", dwell_s=0.0),
            Step(2, "Interlock", "interlock", "Arm it.", dwell_s=0.0, safety_critical=True),
            Step(3, "Run", "run", "Run it.", dwell_s=0.0),
        ),
        out_of_sequence_cooldown_s=0.0,
        idle_reminder_s=1e9,
    )
    from aegis.protocol.engine import Observation

    with tempfile.TemporaryDirectory() as tmp:
        book = Logbook(Path(tmp), "sess_crit", protocol.experiment, len(protocol))
        engine = ProtocolEngine(protocol)
        for e in engine.start():
            book.write_event(e)
        for e in engine.observe(Observation("prep", 0.9)):
            book.write_event(e)
        for e in engine.observe(Observation("run", 0.9)):    # bypasses the interlock
            book.write_event(e)
        report = book.finalise(engine.snapshot(), {})

        text = report.read_text(encoding="utf-8")
        assert "FAIL" in text, "a bypassed safety-critical step must fail the run"
        assert "BLOCKED" in text
        assert "DEVIATIONS AND ALERTS" in text
        assert "none - protocol executed in order" not in text


def test_logbook_elapsed_format_is_sortable():
    assert Logbook._fmt_elapsed(0.0) == "00:00:00.000"
    assert Logbook._fmt_elapsed(65.5) == "00:01:05.500"
    assert Logbook._fmt_elapsed(3725.25) == "01:02:05.250"
