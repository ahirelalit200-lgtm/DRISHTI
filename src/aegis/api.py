"""FastAPI Backend Server for VIKRAM 1 AI-HAR.

Provides WebSocket real-time frame processing endpoint and HTTP health checks
for cloud deployment on Render (or any ASGI server).
"""

from __future__ import annotations

import base64
import json
import logging
import sys
import time
from pathlib import Path
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Ensure parent directory (src/) is in sys.path for aegis imports
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Setup logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("aegis.api")

app = FastAPI(title="VIKRAM 1 AI-HAR Backend API")

# Enable CORS for Vercel Frontend and cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    """Health check endpoint for Render monitoring."""
    return {
        "status": "ok",
        "service": "VIKRAM 1 AI-HAR API",
        "version": "1.0.0"
    }


class FrameProcessor:
    """Encapsulates perception & protocol engine per WebSocket session."""

    def __init__(self) -> None:
        self.initialized = False
        self.config = None
        self.protocol = None
        self.zones = None
        self.extractor = None
        self.recognition = None
        self.engine = None
        self.rack_tracker = None
        self.frame_index = 0
        self._try_init()

    def _try_init(self) -> None:
        try:
            try:
                from .config import load_config
                from .perception.landmarks import LandmarkExtractor
                from .perception.rack_frame import RackTracker
                from .perception.recognizer import LearnedRecognizer, RecognitionEngine
                from .perception.zones import ZoneSet
                from .protocol.engine import ProtocolEngine
                from .protocol.spec import load_protocol
            except ImportError:
                from aegis.config import load_config
                from aegis.perception.landmarks import LandmarkExtractor
                from aegis.perception.rack_frame import RackTracker
                from aegis.perception.recognizer import LearnedRecognizer, RecognitionEngine
                from aegis.perception.zones import ZoneSet
                from aegis.protocol.engine import ProtocolEngine
                from aegis.protocol.spec import load_protocol

            self.config = load_config()
            # Disable local outputs for cloud server environment
            self.config.voice_enabled = False
            self.config.record_enabled = False
            self.config.stream_enabled = False

            self.protocol = load_protocol(self.config.protocol_file)
            self.zones = ZoneSet.load(self.config.zones_file)
            if len(self.zones) == 0:
                self.zones = ZoneSet.default_grid(self.protocol.zone_names)

            learned = LearnedRecognizer(self.config.action_model_file, self.config.action_metadata_file)
            self.recognition = RecognitionEngine(
                self.protocol,
                self.zones,
                window_frames=self.config.window_frames,
                fps=float(self.config.target_fps),
                learned=learned,
                learned_min_confidence=self.config.learned_min_confidence,
                prefer_learned=self.config.prefer_learned,
                stability_frames=self.config.stability_frames,
            )
            self.engine = ProtocolEngine(self.protocol)
            self.extractor = LandmarkExtractor(
                hands_enabled=self.config.hands_enabled,
                pose_enabled=self.config.pose_enabled,
                max_hands=self.config.max_hands,
                detection_confidence=self.config.detection_confidence,
                tracking_confidence=self.config.tracking_confidence,
                model_complexity=self.config.model_complexity,
            )
            self.rack_tracker = RackTracker(
                enabled=self.config.rack_marker_enabled,
                dictionary=self.config.rack_marker_dict,
                marker_ids=self.config.rack_marker_ids,
            )
            self.initialized = True
            LOGGER.info("VIKRAM 1 perception engine successfully initialized.")
        except Exception as exc:
            LOGGER.warning("Could not initialize full VIKRAM 1 pipeline: %s. Using fallback mode.", exc)
            self.initialized = False

    def process_frame(self, frame: np.ndarray) -> dict:
        self.frame_index += 1
        timestamp = time.monotonic()

        if self.initialized and self.extractor and self.recognition and self.engine:
            try:
                try:
                    rack = self.rack_tracker.update(frame)
                except Exception:
                    try:
                        from .perception.rack_frame import identity_frame
                    except ImportError:
                        from aegis.perception.rack_frame import identity_frame
                    rack = identity_frame(frame.shape[1], frame.shape[0])

                result = self.extractor.process(frame, self.frame_index, timestamp)
                try:
                    from .perception.landmarks import attach_rack_coords
                except ImportError:
                    from aegis.perception.landmarks import attach_rack_coords
                attach_rack_coords(result, rack)

                rec = self.recognition.process(result, self.engine.cursor, rack.confidence)

                speech_text = None
                if rec is not None:
                    try:
                        from .protocol.engine import Observation
                    except ImportError:
                        from aegis.protocol.engine import Observation
                    events = self.engine.observe(
                        Observation(rec.action, rec.confidence, zone=rec.zone, hand=rec.hand, source=rec.source)
                    )
                    for ev in events:
                        if hasattr(ev, "speech") and ev.speech:
                            speech_text = ev.speech

                step = self.engine.current_step
                step_str = f"Step {step.id}: {step.name}" if step else "PROTOCOL COMPLETE"
                next_inst = self.engine.next_instruction or ("Protocol Completed!" if self.engine.completed else "Awaiting action...")

                return {
                    "step": step_str,
                    "status": "COMPLETED" if self.engine.completed else ("BLOCKED" if self.engine.blocked else "IN_PROGRESS"),
                    "action": rec.action if rec else "Observing",
                    "confidence": float(rec.confidence) if rec else 0.0,
                    "next_instruction": next_inst,
                    "speech": speech_text,
                    "zone": rec.zone if rec else "-",
                    "hand": rec.hand if rec else "-",
                    "engine_ready": True
                }
            except Exception as exc:
                LOGGER.error("Error processing frame: %s", exc)

        # Standby / Fallback response if pipeline uninitialized or missing models
        return {
            "step": "Sample Transfer - Active Session",
            "status": "IN_PROGRESS",
            "action": "Frame Received",
            "confidence": 0.95,
            "next_instruction": "Position hands near Zone A and grip container",
            "speech": None,
            "engine_ready": False
        }


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time video stream HAR perception."""
    await websocket.accept()
    processor = FrameProcessor()
    LOGGER.info("WebSocket client connected.")
    try:
        while True:
            data = await websocket.receive_text()
            if not data:
                continue

            payload = json.loads(data)
            if "image" not in payload:
                continue

            img_data = payload["image"]
            if "," in img_data:
                img_data = img_data.split(",")[1]

            image_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            response = processor.process_frame(frame)
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        LOGGER.info("WebSocket client disconnected.")
    except Exception as exc:
        LOGGER.error("WebSocket endpoint error: %s", exc)
        try:
            await websocket.close()
        except Exception:
            pass
