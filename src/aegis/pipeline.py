"""Session orchestrator: capture -> perception -> validation -> outputs.

Threading model
---------------
* capture thread   (in :mod:`aegis.capture.source`) grabs frames, keeps latest
* pipeline thread  (here) does perception + validation + annotation
* recorder thread  writes video
* MJPEG threads    serve clients
* voice thread     speaks

The GUI never blocks on any of these: it polls :meth:`snapshot` and
:meth:`latest_frame`. That means a hung speech engine or a wedged network client
cannot freeze the operator's display, which is exactly the failure mode you do
not want on a payload console.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
from pathlib import Path
import threading
import time
from typing import Callable

import numpy as np

from aegis.capture.recorder import VideoRecorder, new_session_id
from aegis.capture.source import VideoSource
from aegis.capture.streamer import VideoStreamer
from aegis.config import AppConfig
from aegis.outputs.logbook import Logbook
from aegis.outputs.overlay import OverlayState, draw, placeholder
from aegis.outputs.voice import VoiceAnnouncer
from aegis.perception.landmarks import LandmarkExtractor, attach_rack_coords
from aegis.perception.rack_frame import RackTracker, identity_frame
from aegis.perception.recognizer import LearnedRecognizer, RecognitionEngine
from aegis.perception.zones import ZoneSet
from aegis.protocol.engine import Observation, ProtocolEngine, ProtocolEvent, Severity
from aegis.protocol.spec import Protocol, load_protocol

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineStatus:
    running: bool = False
    session_id: str = ""
    camera_ok: bool = False
    camera_desc: str = ""
    camera_error: str = ""
    perception: str = ""
    tier: str = ""
    tier_detail: str = ""
    rack_source: str = "-"
    rack_confidence: float = 0.0
    fps: float = 0.0
    latency_ms: float = 0.0
    frames: int = 0
    zones_calibrated: bool = False
    recording: bool = False
    record_path: str = ""
    streaming: bool = False
    stream_url: str = ""
    stream_clients: int = 0
    voice_ok: bool = False
    voice_error: str = ""
    voice_speaking: bool = False
    voice_status: str = "OFF"
    voice_last_text: str = ""
    voice_pending: int = 0
    voice_worker_alive: bool = False
    voice_last_event: str = ""
    log_path: str = ""
    last_action: str = ""
    last_confidence: float = 0.0
    errors: list[str] = field(default_factory=list)


class HARPipeline:
    """One experiment session, end to end."""

    def __init__(self, config: AppConfig, *, on_event: Callable[[ProtocolEvent], None] | None = None) -> None:
        self.config = config
        self.on_event = on_event
        self.protocol: Protocol = load_protocol(config.protocol_file)

        self.zones = ZoneSet.load(config.zones_file)
        self._zones_calibrated = len(self.zones) > 0
        if not self._zones_calibrated:
            self.zones = ZoneSet.default_grid(self.protocol.zone_names)

        learned = LearnedRecognizer(config.action_model_file, config.action_metadata_file)
        self.recognition = RecognitionEngine(
            self.protocol,
            self.zones,
            window_frames=config.window_frames,
            fps=float(config.target_fps),
            learned=learned,
            learned_min_confidence=config.learned_min_confidence,
            prefer_learned=config.prefer_learned,
            stability_frames=config.stability_frames,
        )
        self.engine = ProtocolEngine(self.protocol)
        self.extractor = LandmarkExtractor(
            hands_enabled=config.hands_enabled,
            pose_enabled=config.pose_enabled,
            max_hands=config.max_hands,
            detection_confidence=config.detection_confidence,
            tracking_confidence=config.tracking_confidence,
            model_complexity=config.model_complexity,
        )
        static_quad = None
        rack_meta = self._load_rack_quad()
        if rack_meta is not None:
            static_quad = rack_meta
        self.rack_tracker = RackTracker(
            enabled=config.rack_marker_enabled,
            dictionary=config.rack_marker_dict,
            marker_ids=config.rack_marker_ids,
            static_quad=static_quad,
        )

        self.source = VideoSource(
            config.resolved_source(),
            width=config.frame_width,
            height=config.frame_height,
            fps=config.target_fps,
            flip_horizontal=config.flip_horizontal,
            loop_file=config.loop_video_file,
        )
        self.recorder = VideoRecorder(config.record_directory, fps=config.record_fps)
        self.streamer = VideoStreamer(
            config.stream_host,
            config.stream_port,
            quality=config.stream_quality,
            push_url=config.stream_push_url,
            fps=config.record_fps,
        )
        self.voice = VoiceAnnouncer(
            rate=config.voice_rate,
            volume=config.voice_volume,
            cooldown_s=config.voice_cooldown_s,
            enabled=config.voice_enabled,
        )
        self.logbook: Logbook | None = None

        self.session_id = ""
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._annotated: np.ndarray | None = None
        self._raw: np.ndarray | None = None
        self._events: list[ProtocolEvent] = []
        self._event_cursor = 0
        self._latencies: deque = deque(maxlen=60)
        self._frame_times: deque = deque(maxlen=60)
        self._frames = 0
        self._errors: list[str] = []
        self._overlay = OverlayState()
        self._commands: deque = deque()

    # ---------------------------------------------------------------- helpers

    def _load_rack_quad(self):
        import json

        path = self.config.zones_file.with_name("rack_quad.json")
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            quad = np.asarray(data["quad"], dtype=np.float32).reshape(4, 2)
            return quad
        except Exception as exc:
            LOGGER.warning("could not read rack quad: %s", exc)
            return None

    def _dispatch(self, events: list[ProtocolEvent]) -> None:
        for event in events:
            with self._lock:
                self._events.append(event)
                if len(self._events) > 500:
                    self._events = self._events[-500:]
            if self.logbook is not None:
                self.logbook.write_event(event)
            if self.config.voice_enabled:
                spoken = self.voice.announce_event(event)
                if spoken and self.logbook is not None and getattr(event, "speech", None):
                    self.logbook.write_note("VOICE", f'"{event.speech}"', status="VOICE")
            if event.severity in (Severity.WARNING, Severity.CRITICAL) or event.kind in (
                "step_completed",
                "session_completed",
            ):
                self._overlay.alert = event.message
                self._overlay.alert_severity = event.severity.value
                self._overlay.alert_until = time.monotonic() + (6.0 if event.severity is Severity.CRITICAL else 3.5)
            if self.on_event is not None:
                try:
                    self.on_event(event)
                except Exception as exc:  # pragma: no cover
                    LOGGER.debug("event callback failed: %s", exc)

    # -------------------------------------------------------------- lifecycle

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> PipelineStatus:
        if self.running:
            return self.snapshot()

        self.session_id = new_session_id()
        self._stop.clear()
        self._errors.clear()
        self._frames = 0
        self._event_cursor = 0
        with self._lock:
            self._events.clear()

        self.logbook = Logbook(
            self.config.log_directory, self.session_id, self.protocol.experiment, len(self.protocol)
        )
        self.logbook.write_note("perception", f"landmarks={self.extractor.describe()}")
        self.logbook.write_note("recognition", f"{self.recognition.tier} :: {self.recognition.tier_detail()}")
        self.logbook.write_note(
            "zones",
            f"{len(self.zones)} zones ({'calibrated' if self._zones_calibrated else 'AUTO-GENERATED - run CALIBRATE.bat'})",
            status="INFO" if self._zones_calibrated else "WARN",
        )

        self.recognition.reset()
        self.engine = ProtocolEngine(self.protocol)

        if self.config.voice_enabled:
            self.voice.reset_voice_state(self.engine.state_version)
            self.voice.start()

        if not self.source.start():
            self._errors.append(self.source.error)
            self.logbook.write_note("camera", self.source.error, status="ERROR")
            LOGGER.error("camera failed: %s", self.source.error)

        if self.config.stream_enabled:
            status = self.streamer.start()
            self.logbook.write_note(
                "stream",
                status.url if status.active else f"failed: {status.error}",
                status="INFO" if status.active else "ERROR",
            )
            if self.config.stream_target_ip:
                self.logbook.write_note("stream", f"target ground station: {self.config.stream_target_ip}")

        self._dispatch(self.engine.start())

        self._thread = threading.Thread(target=self._run, name="har-pipeline", daemon=True)
        self._thread.start()
        return self.snapshot()

    def _run(self) -> None:
        recorder_started = False
        while not self._stop.is_set():
            frame = self.source.read(timeout=0.2)
            if frame is None:
                time.sleep(0.005)
                continue

            t0 = time.monotonic()
            image = frame.image

            if self.config.record_enabled and not recorder_started:
                path = self.recorder.start(self.session_id, image.shape[:2])
                recorder_started = True
                if self.logbook is not None:
                    self.logbook.write_note(
                        "recording",
                        str(path) if path else f"failed: {self.recorder.status().error}",
                        status="INFO" if path else "ERROR",
                    )

            try:
                rack = self.rack_tracker.update(image)
            except Exception as exc:  # pragma: no cover
                LOGGER.debug("rack tracking failed: %s", exc)
                rack = identity_frame(image.shape[1], image.shape[0])

            try:
                result = self.extractor.process(image, frame.index, frame.timestamp)
                attach_rack_coords(result, rack)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("perception failed: %s", exc)
                self._note_error(f"perception: {exc}")
                time.sleep(0.02)
                continue

            self._drain_commands()

            try:
                rec = self.recognition.process(result, self.engine.cursor, rack.confidence)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("recognition failed: %s", exc)
                self._note_error(f"recognition: {exc}")
                rec = None

            if rec is not None:
                events = self.engine.observe(
                    Observation(rec.action, rec.confidence, zone=rec.zone, hand=rec.hand, source=rec.source)
                )
                if events:
                    self._dispatch(events)

            self._update_overlay(rec, rack)
            annotated = draw(image, result, rack, self.zones, self._overlay)

            with self._frame_lock:
                self._annotated = annotated
                self._raw = image

            if self.recorder.active:
                self.recorder.write(annotated if self.config.record_annotated else image)
            if self.streamer.active:
                self.streamer.publish(annotated)

            elapsed = time.monotonic() - t0
            self._latencies.append(elapsed * 1000.0)
            self._frame_times.append(time.monotonic())
            self._frames += 1

            if self.engine.completed:
                if self.logbook is not None:
                    self.logbook.write_note("session", "protocol complete - continuing to observe")
                # Keep streaming/recording; the operator decides when to stop.

    def _note_error(self, message: str) -> None:
        with self._lock:
            if message not in self._errors:
                self._errors.append(message)
                if len(self._errors) > 20:
                    self._errors = self._errors[-20:]

    def _update_overlay(self, rec, rack) -> None:
        step = self.engine.current_step
        done, total = self.engine.progress()
        self._overlay.instruction = self.engine.next_instruction
        self._overlay.step_label = (
            f"BLOCKED - STEP {step.id}" if (self.engine.blocked and step) else
            (f"STEP {step.id} / {total} - {step.name}" if step else "PROTOCOL COMPLETE")
        )
        self._overlay.progress = (done, total)
        self._overlay.tier = self.recognition.tier
        self._overlay.fps = self.measured_fps()
        self._overlay.confidence = rec.confidence if rec else 0.0
        self._overlay.rack_source = rack.source
        self._overlay.blocked = self.engine.blocked
        self._overlay.session_id = self.session_id
        self._overlay.recording = self.recorder.active
        self._overlay.streaming = self.streamer.active

    # ---------------------------------------------------------------- control

    def _drain_commands(self) -> None:
        while self._commands:
            try:
                command = self._commands.popleft()
            except IndexError:
                break
            if command == "acknowledge":
                self._dispatch(self.engine.acknowledge())
            elif command == "skip":
                self._dispatch(self.engine.manual_skip())
            elif command == "confirm":
                self._dispatch(self.engine.force_complete())
            elif command == "reset":
                self.engine = ProtocolEngine(self.protocol)
                self.recognition.reset()
                self._dispatch(self.engine.start())

    def acknowledge(self) -> None:
        self._commands.append("acknowledge")

    def manual_skip(self) -> None:
        self._commands.append("skip")

    def confirm_step(self) -> None:
        self._commands.append("confirm")

    def reset_protocol(self) -> None:
        self._commands.append("reset")

    def toggle_recording(self) -> bool:
        if self.recorder.active:
            status = self.recorder.stop()
            if self.logbook:
                self.logbook.write_note("recording", f"stopped: {status.frames} frames -> {status.path}")
            return False
        with self._frame_lock:
            shape = self._raw.shape[:2] if self._raw is not None else (self.config.frame_height, self.config.frame_width)
        path = self.recorder.start(f"{self.session_id}_part{int(time.time())%10000}", shape)
        if self.logbook:
            self.logbook.write_note("recording", f"started: {path}")
        return path is not None

    def toggle_streaming(self) -> bool:
        if self.streamer.active:
            self.streamer.stop()
            if self.logbook:
                self.logbook.write_note("stream", "stopped")
            return False
        status = self.streamer.start()
        if self.logbook:
            self.logbook.write_note("stream", status.url if status.active else f"failed: {status.error}")
        return status.active

    def toggle_voice(self) -> bool:
        self.voice.enabled = not self.voice.enabled and self.voice.available
        if self.voice.enabled:
            self.voice.start()
            self.voice.say("Voice guidance enabled.", force=True)
        return self.voice.enabled

    # ---------------------------------------------------------------- readout

    def measured_fps(self) -> float:
        times = list(self._frame_times)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 1e-6 else 0.0

    def latest_frame(self) -> np.ndarray:
        with self._frame_lock:
            if self._annotated is not None:
                return self._annotated
        message = "CAMERA UNAVAILABLE" if self.source.error else "WAITING FOR SIGNAL"
        return placeholder(self.config.frame_width // 2, self.config.frame_height // 2, message)

    def drain_events(self) -> list[ProtocolEvent]:
        """Events not yet consumed by the GUI."""
        with self._lock:
            new = self._events[self._event_cursor :]
            self._event_cursor = len(self._events)
        return new

    def snapshot(self) -> PipelineStatus:
        stream = self.streamer.status()
        rec_status = self.recorder.status()
        latency = float(np.mean(self._latencies)) if self._latencies else 0.0
        rec = self.recognition.last_raw
        with self._lock:
            errors = list(self._errors)
        return PipelineStatus(
            running=self.running,
            session_id=self.session_id,
            camera_ok=self.source.opened,
            camera_desc=self.source.describe(),
            camera_error=self.source.error,
            perception=self.extractor.describe(),
            tier=self.recognition.tier,
            tier_detail=self.recognition.tier_detail(),
            rack_source=self._overlay.rack_source or "-",
            rack_confidence=0.0,
            fps=self.measured_fps(),
            latency_ms=latency,
            frames=self._frames,
            zones_calibrated=self._zones_calibrated,
            recording=rec_status.active,
            record_path=rec_status.path,
            streaming=stream.active,
            stream_url=stream.url,
            stream_clients=stream.clients,
            voice_ok=self.voice.enabled and self.voice.available,
            voice_error=self.voice.error,
            voice_speaking=self.voice.is_speaking,
            voice_status=self.voice.status_text(),
            voice_last_text=self.voice.last_text,
            voice_pending=self.voice.pending(),
            voice_worker_alive=self.voice.worker_alive,
            voice_last_event=self.voice.last_event_kind,
            log_path=str(self.logbook.text_path) if self.logbook else "",
            last_action=(rec.action or "-") if rec else "-",
            last_confidence=(rec.confidence if rec else 0.0),
            errors=errors,
        )

    def protocol_snapshot(self) -> dict:
        return self.engine.snapshot()

    # ------------------------------------------------------------------ stop

    def stop(self) -> Path | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=4.0)
        self._thread = None

        rec_status = self.recorder.stop()
        self.streamer.stop()
        self.source.stop()

        report: Path | None = None
        if self.logbook is not None:
            extras = {
                "Frames": self._frames,
                "Mean latency": f"{float(np.mean(self._latencies)) if self._latencies else 0:.1f} ms",
                "Recognition": f"{self.recognition.tier}",
                "Video": rec_status.path or "not recorded",
                "Voice alerts": self.voice.spoken_count,
            }
            report = self.logbook.finalise(self.engine.snapshot(), extras)
            self.logbook = None

        self.voice.reset_voice_state()
        self.voice.stop()
        try:
            self.extractor.close()
        except Exception:
            pass
        return report
