# Architecture

Design decisions and the reasoning behind them.

---

## Dataflow

```
  payload camera
        │
        ▼
  ┌───────────────┐   drop-oldest, keeps only the newest frame
  │ VideoSource   │   own thread · auto-reconnect
  └───────┬───────┘
          ▼
  ┌───────────────┐   ArUco → held ArUco → static quad → identity
  │ RackTracker   │   returns a homography: pixels → rack coords
  └───────┬───────┘
          ▼
  ┌───────────────┐   MediaPipe solutions | tasks | motion fallback
  │ LandmarkExtr. │   21 hand pts × 2, 33 pose pts
  └───────┬───────┘
          ▼
  ┌───────────────┐   everything projected into RACK space
  │ FeatureExtr.  │   ~90 relational floats per frame
  └───────┬───────┘
          ▼
  ┌───────────────────────────────┐
  │ RecognitionEngine             │
  │  ├ HeuristicRecognizer  (T0)  │  zones + grip + dwell
  │  ├ LearnedRecognizer    (T1)  │  ONNX GRU over 32-frame window
  │  └ StabilityFilter            │  N consecutive agreeing frames
  └───────┬───────────────────────┘
          ▼  Observation(action, confidence, zone, hand, source)
  ┌───────────────┐   PURE state machine — no I/O, fully unit-tested
  │ ProtocolEngine│   emits ProtocolEvent
  └───────┬───────┘
          ▼
   ┌──────┴──────┬──────────┬──────────┬──────────┐
   ▼             ▼          ▼          ▼          ▼
 Voice        Logbook    Overlay    Recorder   Streamer
 (offline)   (3 files)   (HUD)      (.mp4)     (MJPEG)
                                        │          │
                                        └────┬─────┘
                                             ▼
                                        GUI (polls)
```

---

## Threading model

| Thread | Owns | Failure mode if it stalls |
|---|---|---|
| capture | camera handle | frames age — mitigated by drop-oldest |
| pipeline | perception, validation, annotation | FPS drops; nothing else blocks |
| recorder | video writer | recording drops frames; perception unaffected |
| MJPEG (per client) | one socket | that client stutters; nobody else notices |
| voice | speech engine | alerts queue; no visual impact |
| Tk main | GUI only | — |

**The GUI owns no perception state.** It polls `snapshot()` and `latest_frame()`
on a timer. A hung speech engine or a wedged network client cannot freeze the
operator's display, which is the failure you least want on a payload console.

---

## Key decisions

### Landmarks, not pixels

A 21-point hand plus a 33-point body is ~160 numbers describing the operator,
invariant to lighting, skin tone, sleeve colour and camera gain.

Training a temporal model on *that* needs hundreds of clips, not hundreds of
thousands. This is the single decision that makes a custom experiment protocol
trainable inside a hackathon.

### Rack-relative everything

Features are *relational within the rack frame* — distances, angles, ratios.
Never an absolute image coordinate.

| Consequence | Why it matters |
|---|---|
| Orientation-agnostic | No gravity assumption anywhere. Directly addresses the microgravity clause |
| Camera-agnostic | Move the camera, re-calibrate the quad, the model still works |
| Small | ~90 floats vs a 1280×720 image |

### The capability ladder

Gating everything behind a custom-trained detector means nothing works until the
dataset is done. Instead, three tiers that each run:

* **Tier 0** — rules derived from the protocol YAML. Zero training.
* **Tier 1** — GRU over feature windows. ~30 min of clips.
* **Tier 2** — object detection. Optional.

Tier 1 overrides Tier 0 only above `learned_min_confidence`. The active tier is
displayed, so evidence provenance is always visible.

### The protocol engine is pure

Takes observations and a timestamp, returns events. No camera, no audio, no disk,
injectable clock.

That is why the safety-critical logic is *unit-tested* rather than merely
demonstrated. Thirteen tests cover skip detection, out-of-sequence repeats,
safety blocking, recovery, dwell gating and timeouts — none needing hardware.

### Freshness over completeness

Capture, recording and streaming all drop frames under load rather than queue
them. A warning that arrives late is worse than one computed from a slightly
older frame.

### Explicit degradation

Every fallback — motion-only perception, uncalibrated zones, held rack frame,
missing model, failed codec — is surfaced in the GUI status lamps and written to
the session log. The system never pretends to be more capable than it is.

---

## The state machine

```
              ┌──────────┐
    start ───▶│  ARMED   │◀──────────────┐
              └────┬─────┘               │
                   │ expected action     │ next step armed
                   │ + confidence ≥ bar  │
                   │ + zone matches      │
                   │ + dwell satisfied   │
                   ▼                     │
              ┌──────────┐               │
              │   DONE   │───────────────┘
              └──────────┘

  action belongs to a LATER step:
     ├── no safety-critical step bypassed ──▶ SKIPPED + warn, continue
     └── safety-critical bypassed ──────────▶ BLOCKED
                                              cursor rewinds
                                              all other actions refused
                                              cleared by performing the step
                                              (or operator ACKNOWLEDGE)

  action belongs to an EARLIER completed step ─▶ OUT_OF_SEQUENCE, cursor holds
  action not in the protocol at all ──────────▶ UNEXPECTED_ACTION
  no action, timeout exceeded ────────────────▶ TIMEOUT nudge (once)
```

`BLOCKED` is a distinct state from `FAILED` so the report can distinguish "was
bypassed and recovered" from "was never done".

---

## Extension points

| Want to | Do this |
|---|---|
| Different experiment | Edit `configs/protocol.yaml`. No code changes |
| Different camera | `video_source` accepts index, file, or `rtsp://` |
| Add object detection | Implement `perception/objects.py`, set `objects_enabled` |
| Add HMR | New module producing rack-relative body features; append to the feature vector, bump `FEATURE_VERSION`, retrain |
| Different alert channel | Subscribe to `ProtocolEvent` via the `on_event` callback |
| Different UI | The pipeline is UI-agnostic: `snapshot()`, `protocol_snapshot()`, `latest_frame()`, `drain_events()` |

Bumping `FEATURE_VERSION` is the interlock: the runtime refuses to load a model
trained against a different layout, which prevents confident nonsense after a
recalibration.
