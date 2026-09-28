"""Dataset capture: record labelled clips of your own experiment.

This is the tool that turns the SIH requirement "teams have to build a custom,
highly focused local dataset (even just using a webcam)" into about half an hour
of actual work.

For each clip it stores the extracted **feature sequence**, not the video. That
is the whole trick: a clip becomes a ``(frames, ~90)`` float array of
rack-relative measurements, roughly 30 KB. Two hundred clips is 6 MB, trains in
minutes, and contains no identifiable imagery -- which also sidesteps the
consent problem of shipping a dataset of your teammates' faces.

Optionally saves the raw video alongside (``--keep-video``) so you can re-extract
later if you change the feature layout.

Recommended recipe for a solid model
    * every action label in your protocol, plus ``idle``
    * 12-20 clips per label
    * at least 3 different operators
    * vary: camera distance, lighting, sleeve colour, hand used, speed
    * record at least one full session per operator for the ``idle`` class
    * keep one whole operator out of training -- that is your honest test set

Controls
    1..9, 0     select the action label to record
    SPACE       start / stop recording the current clip
    d           delete the clip you just recorded
    i           jump to the 'idle' label
    TAB         cycle labels
    q / ESC     finish
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

from aegis.capture.source import VideoSource
from aegis.config import load_config
from aegis.perception.features import FEATURE_VERSION, FeatureExtractor
from aegis.perception.landmarks import LandmarkExtractor, attach_rack_coords
from aegis.perception.rack_frame import RackTracker, frame_from_quad
from aegis.perception.zones import ZoneSet
from aegis.protocol.spec import load_protocol

LOGGER = logging.getLogger(__name__)

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


IDLE_LABEL = "idle"


def _load_rack_quad(config):
    path = config.zones_file.with_name("rack_quad.json")
    if not path.exists():
        return None
    try:
        return np.asarray(json.loads(path.read_text(encoding="utf-8"))["quad"], dtype=np.float32)
    except Exception:
        return None


def _panel(img, lines):
    height, width = img.shape[:2]
    box = 24 * len(lines) + 14
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (width, box), (10, 12, 18), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(img, text, (12, 26 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (12, 26 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)


class DatasetRecorder:
    def __init__(self, config, operator: str, session: str, out_dir: Path, keep_video: bool = False) -> None:
        self.config = config
        self.operator = operator
        self.session = session
        self.out_dir = out_dir
        self.keep_video = keep_video
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "clips").mkdir(exist_ok=True)
        if keep_video:
            (self.out_dir / "video").mkdir(exist_ok=True)

        self.protocol = load_protocol(config.protocol_file)
        self.labels = list(self.protocol.action_labels) + [IDLE_LABEL]

        self.zones = ZoneSet.load(config.zones_file)
        self.zones_calibrated = len(self.zones) > 0
        if not self.zones_calibrated:
            self.zones = ZoneSet.default_grid(self.protocol.zone_names)

        self.extractor = LandmarkExtractor(
            hands_enabled=config.hands_enabled,
            pose_enabled=config.pose_enabled,
            max_hands=config.max_hands,
            detection_confidence=config.detection_confidence,
            tracking_confidence=config.tracking_confidence,
            model_complexity=config.model_complexity,
        )
        self.features = FeatureExtractor(self.zones, fps=float(config.target_fps))
        self.rack = RackTracker(
            enabled=config.rack_marker_enabled,
            dictionary=config.rack_marker_dict,
            marker_ids=config.rack_marker_ids,
            static_quad=_load_rack_quad(config),
        )
        self.counts = self._existing_counts()

    def _existing_counts(self) -> dict[str, int]:
        counts = {label: 0 for label in self.labels}
        for path in (self.out_dir / "clips").glob("*.npz"):
            label = path.stem.split("__")[0]
            counts[label] = counts.get(label, 0) + 1
        return counts

    def save_clip(self, label: str, frames: list[np.ndarray], video_frames: list[np.ndarray] | None) -> Path:
        stamp = datetime.now().strftime("%H%M%S")
        stem = f"{label}__{self.operator}__{self.session}__{stamp}"
        path = self.out_dir / "clips" / f"{stem}.npz"
        np.savez_compressed(
            path,
            features=np.asarray(frames, dtype=np.float32),
            label=label,
            operator=self.operator,
            session=self.session,
            feature_version=FEATURE_VERSION,
            dimension=self.features.dimension,
            zone_names=np.array(self.zones.names, dtype=object),
            feature_names=np.array(self.features.feature_names(), dtype=object),
        )
        if video_frames and self.keep_video and cv2 is not None:
            h, w = video_frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(self.out_dir / "video" / f"{stem}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), float(self.config.target_fps), (w, h),
            )
            for f in video_frames:
                writer.write(f)
            writer.release()
        self.counts[label] = self.counts.get(label, 0) + 1
        return path

    def write_manifest(self) -> Path:
        clips = sorted((self.out_dir / "clips").glob("*.npz"))
        rows = []
        for path in clips:
            parts = path.stem.split("__")
            if len(parts) < 4:
                continue
            rows.append({"file": path.name, "label": parts[0], "operator": parts[1], "session": parts[2]})
        manifest = self.out_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "feature_version": FEATURE_VERSION,
                    "dimension": self.features.dimension,
                    "zone_names": self.zones.names,
                    "zones_calibrated": self.zones_calibrated,
                    "labels": self.labels,
                    "clips": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return manifest


def record(config_path: str | None, operator: str, out_dir: str, keep_video: bool, source_override) -> int:
    if cv2 is None:
        print("ERROR: OpenCV is not installed. Run INSTALL.bat first.")
        return 2

    config = load_config(config_path)
    session = datetime.now().strftime("%Y%m%d_%H%M")
    recorder = DatasetRecorder(config, operator, session, Path(out_dir), keep_video)

    if not recorder.zones_calibrated:
        print("WARNING: zones are auto-generated. Run CALIBRATE_ZONES.bat first for a usable model.")
    if recorder.extractor.degraded:
        print(f"WARNING: landmark backend degraded ({recorder.extractor.reason}).")
        print("Recorded features will be near-useless for training. Fix MediaPipe first.")

    source = VideoSource(
        source_override if source_override is not None else config.resolved_source(),
        width=config.frame_width, height=config.frame_height, fps=config.target_fps,
        flip_horizontal=config.flip_horizontal, loop_file=True,
    )
    if not source.start():
        print(f"ERROR: {source.error}")
        return 2

    window = "AEGIS Dataset Recorder"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1180, 720)

    labels = recorder.labels
    label_index = 0
    recording = False
    buffer: list[np.ndarray] = []
    video_buffer: list[np.ndarray] = []
    last_saved: Path | None = None
    started_at = 0.0
    frame_index = 0

    print(__doc__)
    print(f"Labels: {', '.join(f'[{i+1}] {l}' for i, l in enumerate(labels))}")
    print(f"Output: {recorder.out_dir.resolve()}\n")

    while True:
        frame = source.read(timeout=0.1)
        if frame is None:
            continue
        frame_index += 1
        image = frame.image

        rack = recorder.rack.update(image)
        result = recorder.extractor.process(image, frame_index, frame.timestamp)
        attach_rack_coords(result, rack)
        vector = recorder.features.extract(result, rack.confidence)

        if recording:
            buffer.append(vector)
            if keep_video:
                video_buffer.append(image.copy())

        img = image.copy()
        for hand in result.hands:
            pts = hand.points_px.astype(int)
            for p in pts:
                cv2.circle(img, tuple(p), 3, (140, 230, 160), -1, cv2.LINE_AA)
        if rack.corners_px is not None and rack.source != "identity":
            cv2.polylines(img, [rack.corners_px.astype(np.int32).reshape(-1, 1, 2)],
                          True, (120, 220, 140), 1, cv2.LINE_AA)
        for zone in recorder.zones:
            try:
                pts = rack.to_pixels(zone.polygon).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], True, zone.colour, 1, cv2.LINE_AA)
            except Exception:
                pass

        current = labels[label_index]
        tally = "  ".join(f"{l}:{recorder.counts.get(l, 0)}" for l in labels)
        duration = (time.monotonic() - started_at) if recording else 0.0
        _panel(img, [
            (f"LABEL [{label_index + 1}]  {current.upper()}", (240, 220, 150)),
            (f"{'RECORDING ' + format(duration, '.1f') + 's  (' + str(len(buffer)) + ' frames)' if recording else 'SPACE to record'}"
             f"   |   hands: {len(result.hands)}   rack: {rack.source}",
             (90, 90, 250) if recording else (200, 220, 240)),
            (f"counts  {tally}", (140, 165, 195)),
            ("1-9/0 label   TAB cycle   i idle   SPACE rec   d delete last   q finish", (120, 145, 175)),
        ])
        if recording:
            cv2.circle(img, (img.shape[1] - 40, 30), 12, (60, 60, 240), -1, cv2.LINE_AA)

        cv2.imshow(window, img)
        key = cv2.waitKey(15) & 0xFF

        if key in (ord("q"), 27):
            if recording and len(buffer) >= 5:
                last_saved = recorder.save_clip(current, buffer, video_buffer)
                print(f"saved (auto) {last_saved.name}  [{len(buffer)} frames]")
            break

        if key == ord(" "):
            if recording:
                recording = False
                if len(buffer) >= 5:
                    last_saved = recorder.save_clip(current, buffer, video_buffer)
                    print(f"saved {last_saved.name}  [{len(buffer)} frames, {duration:.1f}s]")
                else:
                    print("clip too short (<5 frames) - discarded")
                buffer, video_buffer = [], []
            else:
                recording = True
                started_at = time.monotonic()
                buffer, video_buffer = [], []
                recorder.features.reset()
            continue

        if key == ord("d") and last_saved is not None and last_saved.exists():
            label = last_saved.stem.split("__")[0]
            last_saved.unlink()
            recorder.counts[label] = max(0, recorder.counts.get(label, 1) - 1)
            print(f"deleted {last_saved.name}")
            last_saved = None
            continue

        if key == ord("i"):
            label_index = labels.index(IDLE_LABEL)
            continue

        if key == 9:  # TAB
            label_index = (label_index + 1) % len(labels)
            continue

        if ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(labels):
                label_index = idx
            continue
        if key == ord("0") and len(labels) >= 10:
            label_index = 9
            continue

    source.stop()
    cv2.destroyAllWindows()
    recorder.extractor.close()

    manifest = recorder.write_manifest()
    total = sum(recorder.counts.values())
    print()
    print("=" * 62)
    print(f"Dataset: {recorder.out_dir.resolve()}")
    print(f"Clips  : {total}")
    for label in labels:
        count = recorder.counts.get(label, 0)
        flag = "" if count >= 8 else "   <-- needs more (aim for 12+)"
        print(f"  {label:<24} {count:>3}{flag}")
    print(f"Manifest: {manifest}")
    print("=" * 62)
    print("Next: run TRAIN_MODEL.bat")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a labelled action dataset")
    parser.add_argument("--config", default=None)
    parser.add_argument("--operator", default="op1", help="operator id; keep one operator out of training")
    parser.add_argument("--out", default="datasets/actions", help="dataset directory")
    parser.add_argument("--keep-video", action="store_true", help="also store raw clips as mp4")
    parser.add_argument("--source", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    source = args.source
    if source is not None and str(source).isdigit():
        source = int(source)
    try:
        return record(args.config, args.operator, args.out, args.keep_video, source)
    except KeyboardInterrupt:
        print("\nRecording cancelled.")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
