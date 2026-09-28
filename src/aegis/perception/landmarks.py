"""Pose + hand landmark extraction, with three interchangeable backends.

Why landmarks rather than raw pixels: a 21-point hand skeleton plus a 33-point
body skeleton is a ~160-number description of the operator that is invariant to
lighting, skin tone, sleeve colour and camera gain. Training a temporal model on
*that* needs a few hundred clips, not a few hundred thousand -- which is the only
way a custom experiment protocol becomes trainable inside a hackathon.

Backend selection (automatic, in order):

1. ``solutions``  -- MediaPipe's classic ``mp.solutions.hands`` / ``.pose``.
   Present in the standard PyPI wheels up to 0.10.x. No model files needed.
2. ``tasks``      -- MediaPipe Tasks (``HandLandmarker`` / ``PoseLandmarker``).
   The current supported API; some newer or slimmed wheels ship *only* this.
   Needs ``.task`` bundles on disk, which suits an offline deliverable because
   we ship them with the app (see ``FETCH_MODELS.bat``).
3. ``motion``     -- OpenCV frame differencing. Not real HAR, but it keeps the
   GUI, protocol engine, logging, streaming and recording paths alive on a
   machine where neither MediaPipe path installs.

Discovering which backend a machine actually has, at runtime, is the difference
between "works on my laptop" and "works on the evaluation laptop".
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None  # type: ignore


HAND_LANDMARKS = 21
POSE_LANDMARKS = 33
FINGERTIPS = (4, 8, 12, 16, 20)
WRIST = 0

# Fetched by scripts/fetch_models.py into models/mediapipe/
HAND_TASK = "hand_landmarker.task"
POSE_TASK = "pose_landmarker_lite.task"


@dataclass
class HandObservation:
    """One detected hand, in pixels and in rack-normalised coordinates."""

    label: str                       # "left" | "right"
    score: float
    points_px: np.ndarray            # (21, 2)
    points_rack: np.ndarray | None = None
    world: np.ndarray | None = None  # (21, 3) metric-ish world landmarks

    @property
    def wrist_px(self) -> np.ndarray:
        return self.points_px[WRIST]

    @property
    def centroid_px(self) -> np.ndarray:
        return self.points_px.mean(axis=0)

    def grip_aperture(self) -> float:
        """Mean fingertip-to-wrist distance normalised by hand span.

        Small => closed fist / grasping. Large => open hand / released.
        Scale-free, so it does not change when the operator moves toward the
        camera; rotation-free, so it survives an inverted operator.
        """
        tips = self.points_px[list(FINGERTIPS)]
        wrist = self.points_px[WRIST]
        span = float(np.linalg.norm(self.points_px[5] - self.points_px[17])) + 1e-6
        return float(np.linalg.norm(tips - wrist, axis=1).mean() / span)

    def pinch_distance(self) -> float:
        """Thumb-tip to index-tip distance, normalised by hand span."""
        span = float(np.linalg.norm(self.points_px[5] - self.points_px[17])) + 1e-6
        return float(np.linalg.norm(self.points_px[4] - self.points_px[8]) / span)


@dataclass
class PoseObservation:
    points_px: np.ndarray            # (33, 2)
    visibility: np.ndarray           # (33,)
    world: np.ndarray | None = None  # (33, 3)
    points_rack: np.ndarray | None = None

    def torso_axis(self) -> np.ndarray:
        """Unit vector from hip midpoint to shoulder midpoint (image space).

        This is the operator's own 'up'. Every angle the system computes is
        taken between this axis and a *rack* axis, so nothing references
        gravity: an inverted astronaut produces the same features as an upright
        one performing the same motion.
        """
        shoulders = (self.points_px[11] + self.points_px[12]) / 2.0
        hips = (self.points_px[23] + self.points_px[24]) / 2.0
        vec = shoulders - hips
        return vec / (float(np.linalg.norm(vec)) + 1e-6)


@dataclass
class PerceptionResult:
    frame_index: int
    timestamp: float
    hands: list[HandObservation] = field(default_factory=list)
    pose: PoseObservation | None = None
    objects: list[dict] = field(default_factory=list)
    degraded: bool = False
    notes: str = ""

    def hand(self, label: str) -> HandObservation | None:
        for h in self.hands:
            if h.label == label:
                return h
        return None

    @property
    def has_hands(self) -> bool:
        return bool(self.hands)


def detect_backend(model_dir: Path | None = None) -> str:
    """Report which backend this machine can actually use (for diagnostics)."""
    if mp is None:
        return "motion"
    if hasattr(mp, "solutions"):
        return "solutions"
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
    except Exception:
        return "motion"
    if model_dir is not None and (Path(model_dir) / HAND_TASK).exists():
        return "tasks"
    return "tasks-missing-models"


class LandmarkExtractor:
    """Uniform landmark interface over whichever MediaPipe API is available."""

    def __init__(
        self,
        *,
        hands_enabled: bool = True,
        pose_enabled: bool = True,
        max_hands: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        model_dir: Path | None = None,
    ) -> None:
        default_dir = Path(__file__).resolve().parents[3] / "models" / "mediapipe"
        self.model_dir = Path(model_dir) if model_dir else default_dir
        self.backend = "motion"
        self.hands_enabled = hands_enabled
        self.pose_enabled = pose_enabled
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.model_complexity = model_complexity
        self.reason = ""

        self._hands: Any = None
        self._pose: Any = None
        self._prev_gray: np.ndarray | None = None
        self._ts_ms = 0

        if mp is None:
            self.reason = "mediapipe not installed"
        elif hasattr(mp, "solutions"):
            self._init_solutions()
        else:
            self._init_tasks()

        if self._hands is None and self._pose is None:
            self.backend = "motion"
            self.hands_enabled = self.pose_enabled = False

    # ------------------------------------------------------------- backends

    def _init_solutions(self) -> None:  # pragma: no cover - wheel dependent
        self.backend = "solutions"
        if self.hands_enabled:
            try:
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_hands,
                    model_complexity=min(1, self.model_complexity),
                    min_detection_confidence=self.detection_confidence,
                    min_tracking_confidence=self.tracking_confidence,
                )
            except Exception as exc:
                LOGGER.warning("solutions.Hands failed: %s", exc)
                self.reason = str(exc)
                self.hands_enabled = False
        if self.pose_enabled:
            try:
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=self.model_complexity,
                    smooth_landmarks=True,
                    min_detection_confidence=self.detection_confidence,
                    min_tracking_confidence=self.tracking_confidence,
                )
            except Exception as exc:
                LOGGER.warning("solutions.Pose failed: %s", exc)
                self.reason = str(exc)
                self.pose_enabled = False

    def _init_tasks(self) -> None:  # pragma: no cover - needs .task bundles
        self.backend = "tasks"
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except Exception as exc:
            self.reason = f"mediapipe tasks unavailable: {exc}"
            return

        hand_model = self.model_dir / HAND_TASK
        pose_model = self.model_dir / POSE_TASK

        if self.hands_enabled:
            if not hand_model.exists():
                self.reason = f"missing {hand_model.name} - run FETCH_MODELS.bat"
                self.hands_enabled = False
            else:
                try:
                    options = vision.HandLandmarkerOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=str(hand_model)),
                        running_mode=vision.RunningMode.VIDEO,
                        num_hands=self.max_hands,
                        min_hand_detection_confidence=self.detection_confidence,
                        min_tracking_confidence=self.tracking_confidence,
                    )
                    self._hands = vision.HandLandmarker.create_from_options(options)
                except Exception as exc:
                    LOGGER.warning("HandLandmarker failed: %s", exc)
                    self.reason = str(exc)
                    self.hands_enabled = False

        if self.pose_enabled:
            if not pose_model.exists():
                if not self.reason:
                    self.reason = f"missing {pose_model.name} - run FETCH_MODELS.bat"
                self.pose_enabled = False
            else:
                try:
                    options = vision.PoseLandmarkerOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=str(pose_model)),
                        running_mode=vision.RunningMode.VIDEO,
                        num_poses=1,
                        min_pose_detection_confidence=self.detection_confidence,
                        min_tracking_confidence=self.tracking_confidence,
                    )
                    self._pose = vision.PoseLandmarker.create_from_options(options)
                except Exception as exc:
                    LOGGER.warning("PoseLandmarker failed: %s", exc)
                    self.reason = str(exc)
                    self.pose_enabled = False

    # ------------------------------------------------------------ reporting

    @property
    def degraded(self) -> bool:
        return not (self.hands_enabled or self.pose_enabled)

    def describe(self) -> str:
        if self.degraded:
            return f"motion-only ({self.reason or 'no landmark backend'})"
        parts = [p for p, on in (("hands", self.hands_enabled), ("pose", self.pose_enabled)) if on]
        return f"{self.backend}: {'+'.join(parts)}"

    # ------------------------------------------------------------- inference

    def process(self, frame_bgr: np.ndarray, frame_index: int, timestamp: float) -> PerceptionResult:
        height, width = frame_bgr.shape[:2]
        result = PerceptionResult(frame_index=frame_index, timestamp=timestamp)

        if self.degraded:
            result.degraded = True
            result.notes = "motion-energy fallback"
            blob = self._motion_blob(frame_bgr)
            if blob is not None:
                cx, cy, energy = blob
                pts = self._synthetic_hand(cx, cy, width)
                result.hands.append(HandObservation("right", float(min(1.0, energy)), pts))
            return result

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if cv2 is not None else frame_bgr
        if self.backend == "solutions":
            self._process_solutions(rgb, width, height, result)
        else:
            self._process_tasks(rgb, width, height, result)
        return result

    def _process_solutions(self, rgb, width, height, result) -> None:  # pragma: no cover
        rgb.flags.writeable = False
        if self._hands is not None:
            try:
                out = self._hands.process(rgb)
            except Exception as exc:
                LOGGER.debug("hand inference failed: %s", exc)
                out = None
            if out is not None and out.multi_hand_landmarks:
                handedness = out.multi_handedness or []
                for i, lm in enumerate(out.multi_hand_landmarks):
                    pts = np.array([[p.x * width, p.y * height] for p in lm.landmark], dtype=np.float32)
                    label, score = "right", 0.5
                    if i < len(handedness) and handedness[i].classification:
                        cls = handedness[i].classification[0]
                        label, score = str(cls.label).lower(), float(cls.score)
                    world = None
                    if getattr(out, "multi_hand_world_landmarks", None):
                        try:
                            wl = out.multi_hand_world_landmarks[i]
                            world = np.array([[p.x, p.y, p.z] for p in wl.landmark], dtype=np.float32)
                        except Exception:
                            world = None
                    result.hands.append(HandObservation(label, score, pts, world=world))

        if self._pose is not None:
            try:
                out = self._pose.process(rgb)
            except Exception as exc:
                LOGGER.debug("pose inference failed: %s", exc)
                out = None
            if out is not None and out.pose_landmarks:
                pts = np.array([[p.x * width, p.y * height] for p in out.pose_landmarks.landmark], dtype=np.float32)
                vis = np.array([p.visibility for p in out.pose_landmarks.landmark], dtype=np.float32)
                world = None
                if out.pose_world_landmarks:
                    world = np.array([[p.x, p.y, p.z] for p in out.pose_world_landmarks.landmark], dtype=np.float32)
                result.pose = PoseObservation(pts, vis, world)
        rgb.flags.writeable = True

    def _process_tasks(self, rgb, width, height, result) -> None:  # pragma: no cover
        self._ts_ms += 33
        try:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        except Exception as exc:
            LOGGER.debug("mp.Image failed: %s", exc)
            return

        if self._hands is not None:
            try:
                out = self._hands.detect_for_video(image, self._ts_ms)
            except Exception as exc:
                LOGGER.debug("HandLandmarker failed: %s", exc)
                out = None
            if out is not None and getattr(out, "hand_landmarks", None):
                for i, lm in enumerate(out.hand_landmarks):
                    pts = np.array([[p.x * width, p.y * height] for p in lm], dtype=np.float32)
                    label, score = "right", 0.5
                    try:
                        cat = out.handedness[i][0]
                        label, score = str(cat.category_name).lower(), float(cat.score)
                    except Exception:
                        pass
                    world = None
                    try:
                        wl = out.hand_world_landmarks[i]
                        world = np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float32)
                    except Exception:
                        pass
                    result.hands.append(HandObservation(label, score, pts, world=world))

        if self._pose is not None:
            try:
                out = self._pose.detect_for_video(image, self._ts_ms)
            except Exception as exc:
                LOGGER.debug("PoseLandmarker failed: %s", exc)
                out = None
            if out is not None and getattr(out, "pose_landmarks", None):
                lm = out.pose_landmarks[0]
                pts = np.array([[p.x * width, p.y * height] for p in lm], dtype=np.float32)
                vis = np.array([getattr(p, "visibility", 1.0) for p in lm], dtype=np.float32)
                world = None
                try:
                    wl = out.pose_world_landmarks[0]
                    world = np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float32)
                except Exception:
                    pass
                result.pose = PoseObservation(pts, vis, world)

    # --------------------------------------------------------- degraded mode

    @staticmethod
    def _synthetic_hand(cx: float, cy: float, width: int) -> np.ndarray:
        """A plausible 21-point layout around a motion blob.

        Not a real hand -- it exists purely so downstream code has a consistent
        shape to work with, and so the GUI shows *something* rather than nothing.
        """
        scale = width * 0.045
        angles = np.linspace(-0.9, 0.9, HAND_LANDMARKS)
        radii = np.linspace(0.2, 1.0, HAND_LANDMARKS) * scale
        pts = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1)
        return pts.astype(np.float32)

    def _motion_blob(self, frame_bgr: np.ndarray):  # pragma: no cover
        if cv2 is None:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self._prev_gray is None:
            self._prev_gray = gray
            return None
        delta = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray
        _, thresh = cv2.threshold(delta, 18, 255, cv2.THRESH_BINARY)
        moments = cv2.moments(thresh)
        if moments["m00"] < 500:
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        energy = float(moments["m00"]) / (thresh.size * 255.0) * 20.0
        return cx, cy, energy

    def close(self) -> None:
        for obj in (self._hands, self._pose):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass


def attach_rack_coords(result: PerceptionResult, rack) -> PerceptionResult:
    """Project every landmark into rack-normalised space, in place."""
    for hand in result.hands:
        hand.points_rack = rack.to_rack(hand.points_px)
    if result.pose is not None:
        result.pose.points_rack = rack.to_rack(result.pose.points_px)
    return result
