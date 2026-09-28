"""End-to-end smoke test: real pipeline, synthetic video, no camera required.

Generates a short clip of a moving marker, runs the actual pipeline over it, and
asserts that every required artefact is produced: annotated frames, a session
log, a JSONL machine log, a summary report, and a recorded video file. This is
what proves the plumbing works on a machine with no webcam attached.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

cv2 = pytest.importorskip("cv2")

from aegis.config import AppConfig
from aegis.pipeline import HARPipeline
from aegis.protocol.spec import load_protocol

ROOT = Path(__file__).resolve().parents[1]


def make_clip(path: Path, frames: int = 90, size=(640, 360)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, size)
    assert writer.isOpened(), "could not open test video writer"
    for i in range(frames):
        img = np.full((size[1], size[0], 3), 22, dtype=np.uint8)
        cv2.rectangle(img, (40, 40), (600, 320), (60, 60, 70), 2)
        x = int(120 + 320 * (0.5 + 0.5 * np.sin(i / 12.0)))
        y = int(180 + 60 * np.cos(i / 9.0))
        cv2.circle(img, (x, y), 34, (200, 170, 120), -1)
        writer.write(img)
    writer.release()
    return path


def test_pipeline_produces_all_required_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        clip = make_clip(tmp_path / "clip.mp4")

        cfg = AppConfig(root=ROOT)
        cfg.video_source = str(clip)
        cfg.loop_video_file = True
        cfg.frame_width, cfg.frame_height = 640, 360
        cfg.target_fps = 20
        cfg.record_fps = 20
        cfg.voice_enabled = False           # no audio device in CI
        cfg.record_enabled = True
        cfg.stream_enabled = False
        cfg.log_dir = str(tmp_path / "logs")
        cfg.record_dir = str(tmp_path / "video")
        cfg.protocol_path = str(ROOT / "configs" / "protocol.yaml")
        cfg.zones_path = str(tmp_path / "zones.json")   # absent -> auto grid

        pipeline = HARPipeline(cfg)
        status = pipeline.start()
        assert status.session_id

        deadline = time.time() + 12
        while time.time() < deadline and pipeline.snapshot().frames < 25:
            time.sleep(0.15)

        snap = pipeline.snapshot()
        assert snap.frames >= 25, f"pipeline processed only {snap.frames} frames"
        assert snap.camera_ok, snap.camera_error

        frame = pipeline.latest_frame()
        assert frame.ndim == 3 and frame.shape[2] == 3

        proto = pipeline.protocol_snapshot()
        assert proto["total"] == len(load_protocol(cfg.protocol_file))
        assert proto["next_instruction"]

        report = pipeline.stop()
        assert report is not None and report.exists()

        text = report.read_text(encoding="utf-8")
        assert "EXPERIMENT SESSION REPORT" in text
        assert "Verdict" in text

        logs = sorted((tmp_path / "logs").glob("*.log"))
        jsonl = sorted((tmp_path / "logs").glob("*.jsonl"))
        assert logs and jsonl
        assert "UTC_TIMESTAMP" in logs[0].read_text(encoding="utf-8")

        videos = list((tmp_path / "video").glob("*.mp4")) + list((tmp_path / "video").glob("*.avi"))
        assert videos, "no video was recorded"
        assert videos[0].stat().st_size > 1000


def test_pipeline_survives_a_missing_camera():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = AppConfig(root=ROOT)
        cfg.video_source = str(tmp_path / "does_not_exist.mp4")
        cfg.voice_enabled = False
        cfg.record_enabled = False
        cfg.stream_enabled = False
        cfg.log_dir = str(tmp_path / "logs")
        cfg.protocol_path = str(ROOT / "configs" / "protocol.yaml")
        cfg.zones_path = str(tmp_path / "zones.json")

        pipeline = HARPipeline(cfg)
        status = pipeline.start()
        assert not status.camera_ok
        assert status.camera_error
        # The GUI must still get a frame to display rather than crashing.
        frame = pipeline.latest_frame()
        assert frame.ndim == 3
        report = pipeline.stop()
        assert report is not None and report.exists()
