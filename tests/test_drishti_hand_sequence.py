from aegis.protocol.engine import Observation, ProtocolEngine
from aegis.protocol.spec import load_protocol
from aegis.config import AppConfig

class MockClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, dt: float):
        self.t += dt

def test_hand_observation_sequence_no_false_out_of_sequence():
    config = AppConfig()
    protocol = load_protocol(config.protocol_file)
    clock = MockClock()
    engine = ProtocolEngine(protocol, clock=clock)

    start_events = engine.start()
    assert len(start_events) == 2
    assert engine.cursor == 0

    # Step 1: open_sample_box
    obs1 = Observation(action="open_sample_box", confidence=0.75, zone="outer_box", source="heuristic")
    s1_events = []
    for _ in range(10):
        clock.advance(0.1)
        s1_events.extend(engine.observe(obs1))
    assert engine.cursor == 1
    assert any(e.kind == "step_completed" and e.step_id == 1 for e in s1_events)

    # Step 2: pick_sample
    obs2 = Observation(action="pick_sample", confidence=0.75, zone="outer_box", source="heuristic")
    s2_events = []
    for _ in range(10):
        clock.advance(0.1)
        s2_events.extend(engine.observe(obs2))
    assert engine.cursor == 2
    assert any(e.kind == "step_completed" and e.step_id == 2 for e in s2_events)

    # Step 3: transfer_sample
    obs3 = Observation(action="transfer_sample", confidence=0.78, zone="sample_tray", source="heuristic")
    s3_events = []
    for _ in range(10):
        clock.advance(0.1)
        s3_events.extend(engine.observe(obs3))
    assert engine.cursor == 3
    assert any(e.kind == "step_completed" and e.step_id == 3 for e in s3_events)

    # Hand transit near outer_box while doing Step 4 (Close sample box)
    transit_obs = Observation(action="open_sample_box", confidence=0.65, zone="outer_box", source="heuristic")
    clock.advance(0.1)
    transit_events = engine.observe(transit_obs)
    # Must NOT produce out_of_sequence false alarm!
    assert not any(e.kind == "out_of_sequence" for e in transit_events)

    # Step 4: close_sample_box
    obs4 = Observation(action="close_sample_box", confidence=0.75, zone="outer_box", source="heuristic")
    s4_events = []
    for _ in range(10):
        clock.advance(0.1)
        s4_events.extend(engine.observe(obs4))
    assert engine.cursor == 4
    assert any(e.kind == "step_completed" and e.step_id == 4 for e in s4_events)

    # Step 5: confirm_completion
    obs5 = Observation(action="confirm_completion", confidence=0.75, zone="work_surface", source="heuristic")
    s5_events = []
    for _ in range(10):
        clock.advance(0.1)
        s5_events.extend(engine.observe(obs5))
    assert engine.completed
    assert any(e.kind == "session_completed" for e in s5_events)
