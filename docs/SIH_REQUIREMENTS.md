# Requirements Traceability — SIH Problem Statement 26174

**AI Human Activity Recognition for On-board BAS Experiments**
Indian Space Research Organisation (ISRO) · Department of Space

Every line of the Expected Solution, mapped to the code that implements it, the
test that proves it, and how to demonstrate it live.

---

## Expected Solution

### 1. "The software should continuously process local video feeds to track the sequence of experiment."

| | |
|---|---|
| **Implementation** | `src/aegis/capture/source.py` — threaded grabber, drop-oldest single-slot buffer, automatic reconnect. `src/aegis/pipeline.py` — the perception loop. |
| **Local?** | Yes. No network call anywhere in the inference path. Accepts a USB webcam index, a video file, or a fixed-payload IP camera (`rtsp://`). |
| **Test** | `tests/test_pipeline_smoke.py::test_pipeline_produces_all_required_artifacts` runs the real pipeline over video and asserts ≥25 frames processed. |
| **Demo** | `START.bat` → **START SESSION**. The FPS counter in the metrics bar shows continuous processing. |

Frames are dropped rather than queued when inference is slow. For a system whose
job is to warn *before* the next mistake, a stale-but-complete stream is worse
than a fresh-but-lossy one.

---

### 2. "At the start or after each step, the model should suggest the next step to be performed."

| | |
|---|---|
| **Implementation** | `src/aegis/protocol/engine.py` — `ProtocolEngine._emit_armed()` fires a `step_armed` event on session start and after every completion. |
| **Surfaces** | GUI **NEXT STEP** card (largest text on screen), burned into the video overlay, spoken aloud, and written to the log. |
| **Test** | `test_protocol_engine.py::test_next_instruction_is_always_available` |
| **Demo** | The card updates the instant a step is verified. |

An idle reminder re-announces the instruction after `idle_reminder_s`, so an
operator who looks away does not have to ask what was next.

---

### 3. "It should alert when a step is skipped or an out of sequence step is added. It should be a voice based alert."

Three distinct violation classes, not one:

| Violation | Trigger | Response |
|---|---|---|
| **Skip (non-critical)** | An action belonging to a later step fires | Voice warning naming the skipped step, logged, run continues |
| **Skip (safety-critical)** | Same, but a bypassed step has `safety_critical: true` | **Voice alarm, guidance HALTS**, cursor rewinds to the missed step, red border on video |
| **Out of sequence** | An already-completed step is repeated, or an action outside the protocol | Voice warning, cursor does **not** advance |

| | |
|---|---|
| **Implementation** | `engine.py` — `_handle_forward_jump()`, the `past`/`future` branch in `observe()`. Voice: `src/aegis/outputs/voice.py`. |
| **Voice is offline** | `pyttsx3` → SAPI5 on Windows. No cloud TTS, no internet. |
| **Priority** | Critical utterances pre-empt and flush queued routine prompts, so a safety alarm never waits behind three "next step" prompts. |
| **Dedup** | Per-utterance cooldown; a violation persisting for 90 frames is spoken once, not 90 times. |
| **Tests** | `test_skipping_a_noncritical_step_warns_and_continues`, `test_skipping_a_safety_critical_step_blocks`, `test_block_is_cleared_by_performing_the_missed_step`, `test_repeating_a_completed_step_is_flagged_out_of_sequence`, `test_unknown_action_is_reported_not_ignored` |
| **Demo** | Deliberately perform step 5 before step 4 → hear the alert, see the rail turn amber. Skip a critical step → hear the alarm, watch guidance halt. |

Recovery is designed in: performing the missed critical step clears the block
automatically, and the log records both the bypass and the recovery.

---

### 4. "Using the live video, it should generate a timestamped and structured lightweight text file of the conducted steps with outcomes/status."

Three artefacts per session, all plain text, all flushed line-by-line so a power
loss costs at most one line:

| File | Format | Purpose |
|---|---|---|
| `logs/<session>.log` | Fixed-width columns | Human-readable — what a jury reads |
| `logs/<session>.jsonl` | One JSON object per line | Machine-parseable for downlink |
| `logs/<session>_report.txt` | Summary table | Per-step outcome, verdict, deviations |

Column layout: `UTC_TIMESTAMP | ELAPSED | EVENT | STEP | STATUS | CONF | DETAIL`

| | |
|---|---|
| **Implementation** | `src/aegis/outputs/logbook.py` |
| **Lightweight?** | A full 8-step session is a few kB. No images, no video, no base64. |
| **Outcomes** | Per step: `DONE` / `SKIPPED` / `BLOCKED` / `FAILED` / `PENDING`, plus confidence, duration, evidence source (heuristic / learned / manual), and the `outcome_hint` from the protocol. |
| **Verdict** | `PASS` / `PASS WITH DEVIATIONS` / `FAIL`. A bypassed safety-critical step forces `FAIL`. |
| **Tests** | `test_logbook_writes_all_three_artifacts`, `test_report_verdict_reflects_a_bypassed_critical_step`, `test_logbook_elapsed_format_is_sortable` |
| **Demo** | GUI → **OPEN LOGS**, or the prompt shown on session stop. |

---

### 5. "Stream the video of the experiment to specific IP and also store the video locally."

**Stream** — two modes, because "to a specific IP" means different things
depending on who opens the connection:

| Mode | Mechanism | When to use |
|---|---|---|
| **Pull** (default) | MJPEG over HTTP on `stream_host:stream_port`. Ground station opens `http://<payload-pc>:8090/` | Zero extra software at either end; works in a browser or VLC |
| **Push** (optional) | Frames piped to `ffmpeg` → RTSP/RTMP/UDP at `stream_push_url` | When the ground station must be the receiver |

**Store** — `outputs/recordings/<session>.mp4`, written on a dedicated thread
with a bounded queue. Codec fallback chain `mp4v → avc1 → MJPG/.avi`, so it
never silently fails to record. `record_annotated: true` burns the HUD into the
archive, so the stored video shows what the system believed and when.

| | |
|---|---|
| **Implementation** | `src/aegis/capture/streamer.py`, `src/aegis/capture/recorder.py` |
| **No heavy deps** | MJPEG server is stdlib `http.server` — no FastAPI, no uvicorn |
| **Backpressure** | Only the latest frame is held; a slow client throttles itself instead of stalling perception |
| **Test** | Smoke test asserts a non-empty `.mp4` is produced. `DIAGNOSTICS.bat` verifies port availability and reports the exact URL. |
| **Demo** | GUI → **STREAM**, then open the URL it prints on a phone on the same Wi-Fi. |

---

### 6. "A graphical user interface for monitoring the above activities."

`src/aegis/gui/app.py` — Tkinter, ships with Python, no Qt licensing question.

* Live annotated video with skeletons, rack quad, zones, HUD
* **NEXT STEP** card — deliberately the largest element on screen
* Protocol rail — every step with live state colouring and a `CRIT` badge
* Colour-coded event log
* Six status lamps: camera, model, rack, voice, record, stream
* Metrics: FPS, latency, active recognition tier, current observation + confidence
* Controls: Start, Stop, Acknowledge, Confirm Step, Skip, Record, Stream, Voice, Open Logs

| | |
|---|---|
| **Never freezes** | The GUI owns no perception state; it polls the pipeline on a timer. A hung speech engine or wedged network client cannot lock the console. |
| **Test** | Verified under a virtual display: builds, runs a live pipeline, renders frames, shuts down cleanly. |

---

### 7. "Deliverable: A trained AI model that runs on offline standalone system"

| | |
|---|---|
| **Artefact** | `models/action/action_model.onnx` + `.meta.json` sidecar + auto-generated `MODEL_CARD.md` |
| **Architecture** | GRU over 32-frame windows of ~90 rack-relative features → mean pool → linear head |
| **Runtime** | `onnxruntime` CPU. No internet, no GPU required, no cloud API |
| **Training** | `TRAIN_MODEL.bat`. PyTorch backend if installed; otherwise a pure-NumPy GRU with hand-written backprop — so training needs no 2.5 GB dependency |
| **Validation** | **Operator-held-out**, not random split |
| **Safety interlock** | The runtime refuses to load a model whose feature version or dimension does not match the current zone configuration |
| **Test** | `tests/test_onnx_export.py` — numerical parity between the trained NumPy model and the exported ONNX graph |

**On validation methodology.** A random train/test split leaks badly here:
consecutive sliding windows from one clip are near-duplicates, so a random split
reports ~99% and then collapses on a new person. Holding out an entire operator
is the honest number and the one printed in the model card.

---

## Description clauses

### "Standalone operation... data is processed locally at the 'edge.'"

Nothing in the inference path touches the network. Streaming is *outbound and
optional*. The only component that ever downloads anything is `FETCH_MODELS.bat`,
run once at setup; the fetched bundles ship inside the package.

### "Inputs are given from fixed-payload cameras."

`video_source` accepts a webcam index, a file, or an `rtsp://` URL. The static
rack-quad calibration path exists specifically for the fixed-camera case.

### "Dataset generation to train model for object detection, pose estimation and hand-object interaction"

| Requirement | Implementation |
|---|---|
| Dataset generation | `RECORD_DATASET.bat` — press a number key, press SPACE, perform the action |
| Pose estimation | MediaPipe Pose, 33 landmarks, rack-projected |
| Hand-object interaction | 21-point hand skeletons + grip aperture + pinch distance + calibrated zone occupancy |
| Object detection | Tier 2, optional (`objects_enabled`) — the system does not depend on it |

Clips store the **feature sequence**, not video: ~30 kB each. Two hundred clips
is 6 MB, trains in minutes, and contains no identifiable imagery — which also
avoids shipping a dataset of your teammates' faces.

### Optional: "orientation-agnostic 3D Human Mesh Recovery (HMR) to track the astronaut's body relative to the payload rack, not the floor."

**Addressed, by a route that is lighter and more robust than full HMR.**

Four ArUco markers on the rack give a homography from image space into rack
coordinates, recovered every frame. Every feature the system computes is
*relational within that frame* — distances, angles, ratios between the operator
and the rack. Nothing references gravity or a floor plane.

| | |
|---|---|
| **Implementation** | `src/aegis/perception/rack_frame.py`, `src/aegis/perception/features.py` |
| **Proof** | `tests/test_rack_frame.py::test_rack_coordinates_are_invariant_to_rack_rotation` renders the scene at 0°, 30°, 90°, 180°, 270° and asserts a fixed physical point keeps the same rack coordinates to within 0.06 |
| **Occlusion** | `test_occlusion_holds_the_last_good_frame` — losing markers behind an arm holds the last good frame for a grace period rather than collapsing |
| **Fallback chain** | ArUco → held ArUco → calibrated static quad → whole-image identity |
| **Why not full HMR** | HMR2.0 / 4D-Humans needs ~4 GB VRAM and runs at 5–10 fps on a laptop GPU. It recovers a full body mesh — far more than sequence validation needs — while the *actual* requirement is a rack-relative reference frame, which a planar homography gives exactly, at 200+ fps, on CPU. The architecture leaves a clean seam to add HMR as a Tier-3 module if body-configuration features are ever needed. |

---

## Traceability summary

| # | Requirement | Status | Primary test |
|---|---|---|---|
| 1 | Continuous local video processing | Implemented | `test_pipeline_smoke.py` |
| 2 | Suggest the next step | Implemented | `test_next_instruction_is_always_available` |
| 3 | Voice alert on skip / out-of-sequence | Implemented | 5 dedicated tests |
| 4 | Timestamped structured text log | Implemented | `test_logbook_writes_all_three_artifacts` |
| 5 | Stream to IP + store locally | Implemented | smoke test + diagnostics |
| 6 | GUI | Implemented | headless GUI test |
| 7 | Trained model, offline standalone | Implemented | `test_onnx_export.py` |
| — | Standalone / edge | Implemented | no network in inference path |
| — | Dataset generation | Implemented | `RECORD_DATASET.bat` |
| — | Orientation-agnostic (optional) | Implemented | `test_rack_frame.py` (5 rotations) |

**45 automated tests, all passing, none requiring a camera or GPU.**
Run them yourself: `RUN_TESTS.bat`.

---

## Honest limitations

Worth stating before a jury asks:

1. **Out-of-the-box accuracy is heuristic-grade.** Tier 0 uses zone occupancy,
   grip aperture and dwell. It is genuinely useful and demonstrable on day one,
   but it is not a trained model. Tier 1 requires you to record ~30 minutes of
   clips of *your* experiment.
2. **Actions distinguished only by which object is held** — identical motion,
   identical location, different item — are the weak point. That is precisely
   what the optional Tier-2 object detector is for.
3. **Single camera geometry.** Substantially repositioning the camera means
   recalibrating zones and retraining.
4. **Microgravity is simulated, not tested.** The orientation-invariance is
   proven mathematically and in synthetic rotation tests, not in freefall.
5. **Voice alerts depend on an OS speech engine.** Diagnostics reports if it is
   unavailable; the visual alert path is independent and always works.
