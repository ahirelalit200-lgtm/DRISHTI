<div align="center">

# AEGIS · AI-HAR

**AI Human Activity Recognition for On-board BAS Experiments**

Smart India Hackathon 2026 · Problem Statement **SIH26174** · ISRO / Department of Space

*Recognises and validates the sequence of a pre-defined experiment, on-board,
offline, with voice guidance — so science does not wait for a round trip to Earth.*

</div>

---

## 1. Executive Summary & SIH26174 Mapping

An astronaut performing a scientific protocol has no ground support in real time.
AEGIS watches the payload camera, recognises what the operator is doing using orientation-agnostic rack features, checks it against the expected protocol, tells them what to do next, and issues voice alerts if a step is skipped or performed out of order. Everything runs locally and 100% offline. Every decision is logged with its timestamped evidence.

---

## 2. Features

* **Real-time Perception & Landmark Extraction**: Uses MediaPipe Pose and Hand landmark tracking at 30 FPS.
* **Orientation-Agnostic Rack Frame**: 4 ArUco markers or static quad calibration provide homography into rack coordinates, solving microgravity orientation variance without high-overhead 3D HMR.
* **Dual-Tier Recognition**:
  * **Tier 0 (Heuristic)**: Rack-relative zones + grip aperture transition + dwell gating. Works immediately with zero training.
  * **Tier 1 (Learned)**: GRU model over 32-frame sliding windows of ~90 rack-relative features.
* **Pure Protocol Engine**: State machine supporting sequence verification, dwell gating, step arming, idle reminders, out-of-sequence alerts, skipped-step detection, and safety-critical hard halts.
* **Voice Guidance & Alerts**: Local, offline Text-to-Speech (SAPI5 via `pyttsx3`) issuing step instructions and safety warnings.
* **Mission Console GUI**: Tkinter GUI displaying live annotated video, subsystem status lamps, next-step instruction card, step progress rail, scrolling event log, and manual controls.
* **Comprehensive Logging & Evidence**: Writes fixed-width `.log` files, machine-readable `.jsonl` telemetry, and formatted summary `_report.txt` files.
* **HUD Video Recording & Live MJPEG Streaming**: Burns live status overlays onto saved MP4 videos and streams live MJPEG over local HTTP (port 8095).

---

## 3. Architecture

```
AEGIS-AI-HAR/
├── configs/
│   ├── app.yaml              # Runtime settings & video source
│   ├── protocol.yaml         # Experiment protocol definition (5-step Sample Transfer)
│   ├── zones.json            # Calibrated rack-relative interaction zones
│   └── rack_quad.json        # Static rack corners fallback
├── models/
│   ├── action/               # Trained ONNX action model & metadata
│   └── mediapipe/            # Offline MediaPipe .task bundles
├── logs/                     # Session logs (.log, .jsonl, _report.txt)
├── outputs/
│   └── recordings/           # Session MP4 video recordings
├── src/aegis/
│   ├── capture/              # Video capture, MP4 recorder, MJPEG streamer
│   ├── perception/           # Rack frame homography, landmarks, zones, features, recognizer
│   ├── protocol/             # Protocol spec, state machine engine
│   ├── outputs/              # Voice TTS, logbook recorder, HUD overlay
│   ├── gui/                  # Tkinter mission console app
│   ├── tools/                # Calibrate, diagnostics, fetch_models, train
│   └── pipeline.py           # Core session orchestrator
├── tests/                    # Unit & integration test suite (40 tests)
├── pyproject.toml
└── requirements.txt
```

---

## 4. Requirements & Installation

### Requirements
* Windows 10/11 (Linux and macOS compatible)
* Python 3.12 (64-bit) with `tcl/tk`
* Any standard USB webcam
* 100% offline (no cloud dependencies)

### Quick Setup

```powershell
# 1. Create Python 3.12 Virtual Environment
py -3.12 -m venv .venv

# 2. Install Dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt pytest

# 3. Fetch Offline MediaPipe Models
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m aegis.tools.fetch_models

# 4. Run Environment Diagnostics
.\.venv\Scripts\python.exe -m aegis.tools.diagnostics
```

---

## 5. How to Start

To launch the AEGIS Mission Console GUI:

```powershell
# Windows batch launch
START.bat

# Or direct command:
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m aegis.gui.app
```

---

## 6. Camera Setup

In `configs/app.yaml`:
```yaml
video_source: 0          # 0, 1, 2 for USB webcam, or path to MP4 video file
frame_width: 1280
frame_height: 720
target_fps: 30
```
Or launch with `--source 0` or `--source "datasets/demo.mp4"`.

---

## 7. Demo Experiment: Sample Transfer Experiment

The system is pre-configured with the **SIH26174 Sample Transfer Experiment** protocol in `configs/protocol.yaml`:

| Step ID | Name | Action | Zone | Safety Critical | Instruction |
|:---:|---|---|---|:---:|---|
| **1** | Open sample box | `open_sample_box` | `outer_box` | False | Open the sample box lid. |
| **2** | Pick sample | `pick_sample` | `outer_box` | False | Pick the sample from the sample box. |
| **3** | Transfer sample to tray | `transfer_sample` | `sample_tray` | **True** | Transfer the sample into the sample tray. |
| **4** | Close sample box | `close_sample_box` | `work_surface` | **True** | Close the sample box lid. |
| **5** | Confirm completion | `confirm_completion` | `work_surface` | **True** | Confirm experiment completion on the control panel. |

---

## 8. How to Demonstrate

### Demonstration Setup
1. Launch `START.bat` to open the console.
2. Click **START SESSION**.

### Demonstrating Correct Sequence
Press keyboard hotkeys **1 → 2 → 3 → 4 → 5** (or perform hand actions in camera view):
* **Step 1**: Step 1 turns GREEN (`OK`), Next Step card updates to Step 2, Voice announces: *"Step 1 complete. Next step: Pick sample."*
* **Step 2**: Step 2 turns GREEN (`OK`), Next Step card updates to Step 3, Voice announces: *"Step 2 complete. Next step: Transfer sample to tray."*
* **Step 3**: Step 3 turns GREEN (`OK`), Next Step card updates to Step 4.
* **Step 4**: Step 4 turns GREEN (`OK`), Next Step card updates to Step 5.
* **Step 5**: Experiment reaches 100% completion (`EXPERIMENT COMPLETE`), summary report generated.

### Demonstrating Wrong / Skipped Sequence
Press keyboard hotkeys **1 → 3** (skipping Step 2):
* **Result**: Because Step 3 is safety-critical, the protocol engine triggers a **HARD SAFETY BLOCK**.
* **GUI Banner**: Turns RED: `HALTED - RECOVER STEP 2`.
* **Voice Alert**: *"Warning. Safety critical step 2, Pick sample, was skipped. Stop and complete it before continuing."*
* **Event Log**: Records `CRITICAL | SAFETY BLOCK. Detected 'Transfer sample to tray' but safety-critical step 2 'Pick sample' was not performed.`
* **Recovery**: Press Key **2** (or perform Step 2 action) → Voice announces: *"Good. Safety step recovered."* -> System unblocks and resumes guidance.

---

## 9. Output & Log Locations

Every session automatically records:
* **Human Log**: `logs/<session_id>.log` (timestamped events with severity)
* **Machine Log**: `logs/<session_id>.jsonl` (structured JSON for ground station transmission)
* **Summary Report**: `logs/<session_id>_report.txt` (final verdict, completed steps, safety violations)
* **Annotated Video**: `outputs/recordings/<session_id>.mp4` (HUD burned-in session video)

---

## 10. Limitations & Future Enhancements

### Known Limitations
1. **Tier-0 Heuristic Baseline**: Out of the box, activity recognition uses spatial zone dwell and hand grip trends. Tier 1 requires recording ~30 mins of user clips.
2. **Single Camera Setup**: Designed for standard ISRO payload single-camera view.

### Future Enhancements (P2)
1. **3D Human Mesh Recovery (HMR)**: Full 3D joint reconstruction.
2. **Multi-Camera Fusion**: Dual-camera tracking for deep occlusion resilience.
3. **Edge NPU Acceleration**: Optimization for space-grade hardware (Xilinx Versal / ARM NPU).

---

## 11. SIH26174 Feature Traceability Matrix

| SIH26174 Requirement | AEGIS Implementation | Status |
|---|---|:---:|
| **Continuous camera processing** | Threaded OpenCV `VideoSource` frame pipeline | **IMPLEMENTED** |
| **Human activity recognition** | MediaPipe landmark extraction + Tier-0 Heuristic & Tier-1 GRU Recognizer | **IMPLEMENTED** |
| **Experiment sequence recognition** | `ProtocolEngine` state machine with dwell gating & step arming | **IMPLEMENTED** |
| **Next-step suggestion** | Prominent HUD NEXT STEP card & voice announcer guidance | **IMPLEMENTED** |
| **Skipped-step detection** | Forward jump detection in `ProtocolEngine` engaging safety block | **IMPLEMENTED** |
| **Out-of-sequence detection** | State machine validation rejecting out-of-order action execution | **IMPLEMENTED** |
| **Voice alert** | Local, offline `pyttsx3` Text-to-Speech announcer (SAPI5) | **IMPLEMENTED** |
| **Timestamped experiment record** | Fixed-width `.log` file, machine `.jsonl`, and summary report | **IMPLEMENTED** |
| **Video storage** | Threaded `VideoRecorder` burning HUD annotations to MP4 | **IMPLEMENTED** |
| **Video streaming** | Built-in MJPEG HTTP streamer on port 8095 & optional RTSP push | **IMPLEMENTED** |
| **Mission Console GUI** | Tkinter console with live feed, status lamps, rail, log & hotkeys | **IMPLEMENTED** |
| **Offline operation** | 100% offline execution with local models and zero cloud API calls | **IMPLEMENTED** |
| **3D HMR** | Orientation-agnostic ArUco homography baseline (3D HMR reserved for P2) | **FUTURE / OPTIONAL** |

