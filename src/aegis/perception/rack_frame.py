"""Orientation-agnostic rack-relative coordinate frame.

The SIH statement calls out that ground-based posture models fail in microgravity
because there is no fixed 'up'. The fix is not to guess gravity -- it is to stop
depending on it. Everything the system reasons about (hand positions, zones,
velocities) is expressed in the **payload rack's** own coordinate system, so the
maths is identical whether the operator is upright, sideways, or inverted.

Two ways to establish that frame:

1. **ArUco markers** stuck on the rack (recommended). Four markers give a full
   planar homography, so we recover rack coordinates exactly, at any roll angle.
2. **Static fallback** -- the operator marks the rack corners once during
   calibration. Correct as long as the camera is fixed to the rack, which is the
   payload-camera case described in the problem statement.

In both cases the output is the same: a function mapping image pixels to
normalised rack coordinates in [0, 1] x [0, 1], plus the rack's roll angle in the
image so the HUD can be drawn the right way up.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

try:  # pragma: no cover - exercised only when OpenCV is installed
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


ARUCO_DICTS = {
    "DICT_4X4_50": 0,
    "DICT_4X4_100": 1,
    "DICT_5X5_50": 4,
    "DICT_6X6_250": 10,
    "DICT_APRILTAG_36h11": 20,
}


@dataclass
class RackFrame:
    """A resolved mapping from image space to rack space."""

    homography: np.ndarray | None      # 3x3, pixels -> rack unit square
    inverse: np.ndarray | None         # 3x3, rack unit square -> pixels
    corners_px: np.ndarray | None      # 4x2 image-space rack quad (TL,TR,BR,BL)
    roll_deg: float                    # rack rotation in the image, degrees
    source: str                        # "aruco" | "static" | "identity"
    confidence: float

    @property
    def valid(self) -> bool:
        return self.homography is not None

    def to_rack(self, points_px: np.ndarray) -> np.ndarray:
        """Map an (N,2) array of pixel coords into rack-normalised coords."""
        pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
        if self.homography is None:
            return pts.reshape(-1, 2)
        if cv2 is not None:
            out = cv2.perspectiveTransform(pts, self.homography)
            return out.reshape(-1, 2)
        return _manual_perspective(pts.reshape(-1, 2), self.homography)

    def to_pixels(self, points_rack: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_rack, dtype=np.float32).reshape(-1, 1, 2)
        if self.inverse is None:
            return pts.reshape(-1, 2)
        if cv2 is not None:
            out = cv2.perspectiveTransform(pts, self.inverse)
            return out.reshape(-1, 2)
        return _manual_perspective(pts.reshape(-1, 2), self.inverse)


def _manual_perspective(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homo = np.hstack([points.astype(np.float32), ones])
    out = homo @ matrix.T
    w = np.where(np.abs(out[:, 2:3]) < 1e-9, 1e-9, out[:, 2:3])
    return (out[:, :2] / w).astype(np.float32)


def identity_frame(width: int, height: int) -> RackFrame:
    """Fallback frame: the whole image *is* the rack (normalised pixels)."""
    scale = np.array(
        [[1.0 / max(width, 1), 0.0, 0.0], [0.0, 1.0 / max(height, 1), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return RackFrame(
        homography=scale,
        inverse=np.linalg.inv(scale).astype(np.float32),
        corners_px=np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        ),
        roll_deg=0.0,
        source="identity",
        confidence=0.25,
    )


def frame_from_quad(quad_px: Sequence[Sequence[float]], source: str = "static", confidence: float = 0.8) -> RackFrame:
    """Build a rack frame from four image-space corners, ordered TL,TR,BR,BL."""
    src = np.asarray(quad_px, dtype=np.float32).reshape(4, 2)
    dst = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    if cv2 is not None:
        h = cv2.getPerspectiveTransform(src, dst)
    else:  # pragma: no cover
        h = _dlt(src, dst)
    top_edge = src[1] - src[0]
    roll = math.degrees(math.atan2(float(top_edge[1]), float(top_edge[0])))
    return RackFrame(
        homography=h.astype(np.float32),
        inverse=np.linalg.inv(h).astype(np.float32),
        corners_px=src,
        roll_deg=roll,
        source=source,
        confidence=confidence,
    )


def _dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:  # pragma: no cover
    """Minimal direct linear transform, used only if OpenCV is unavailable."""
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    h = vh[-1].reshape(3, 3)
    return h / h[2, 2]


def order_quad(points: np.ndarray) -> np.ndarray:
    """Order four arbitrary points as TL, TR, BR, BL by angle about the centroid."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    order = np.argsort(angles)
    ordered = pts[order]
    # rotate so the point closest to the top-left of the bounding box comes first
    sums = ordered.sum(axis=1)
    start = int(np.argmin(sums))
    return np.roll(ordered, -start, axis=0)


class RackTracker:
    """Resolves a :class:`RackFrame` for each incoming frame.

    Falls back gracefully: ArUco -> last good ArUco (for a grace period) ->
    calibrated static quad -> whole-image identity. The pipeline therefore never
    stalls just because a marker was occluded by the operator's arm.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        dictionary: str = "DICT_4X4_50",
        marker_ids: Sequence[int] | None = None,
        static_quad: Sequence[Sequence[float]] | None = None,
        grace_frames: int = 45,
    ) -> None:
        self.enabled = enabled and cv2 is not None and hasattr(cv2, "aruco")
        self.marker_ids = list(marker_ids or [0, 1, 2, 3])
        self.static_quad = np.asarray(static_quad, dtype=np.float32) if static_quad is not None else None
        self.grace_frames = grace_frames
        self._misses = 0
        self._last: RackFrame | None = None
        self._detector = None
        if self.enabled:
            self._detector = self._build_detector(dictionary)
            if self._detector is None:
                self.enabled = False

    def _build_detector(self, dictionary: str):  # pragma: no cover - needs cv2.aruco
        try:
            dict_id = getattr(cv2.aruco, dictionary, None)
            if dict_id is None:
                dict_id = getattr(cv2.aruco, "DICT_4X4_50")
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters()
            return cv2.aruco.ArucoDetector(aruco_dict, params)
        except Exception:
            return None

    def _detect_aruco(self, gray: np.ndarray) -> RackFrame | None:  # pragma: no cover
        if self._detector is None:
            return None
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return None
        found: dict[int, np.ndarray] = {}
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            found[int(marker_id)] = marker_corners.reshape(4, 2).mean(axis=0)

        wanted = [m for m in self.marker_ids if m in found]
        if len(wanted) >= 4:
            quad = np.array([found[m] for m in wanted[:4]], dtype=np.float32)
            return frame_from_quad(order_quad(quad), source="aruco", confidence=0.95)
        if len(found) >= 1:
            # A single marker still pins position + roll; assume a rack span of
            # 6x the marker width, which the operator sets during calibration.
            marker_id = wanted[0] if wanted else next(iter(found))
            idx = [int(v) for v in ids.flatten()].index(marker_id)
            pts = corners[idx].reshape(4, 2)
            centre = pts.mean(axis=0)
            edge = pts[1] - pts[0]
            size = float(np.linalg.norm(edge)) * 6.0
            angle = math.atan2(float(edge[1]), float(edge[0]))
            half = size / 2.0
            base = np.array([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=np.float32)
            rot = np.array(
                [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
                dtype=np.float32,
            )
            quad = (base @ rot.T) + centre
            return frame_from_quad(quad, source="aruco", confidence=0.7)
        return None

    def update(self, frame_bgr: np.ndarray) -> RackFrame:
        height, width = frame_bgr.shape[:2]

        if self.enabled and cv2 is not None:
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                resolved = self._detect_aruco(gray)
            except Exception:
                resolved = None
            if resolved is not None:
                self._last = resolved
                self._misses = 0
                return resolved
            self._misses += 1
            if self._last is not None and self._misses <= self.grace_frames:
                held = RackFrame(
                    self._last.homography,
                    self._last.inverse,
                    self._last.corners_px,
                    self._last.roll_deg,
                    source="aruco_hold",
                    confidence=max(0.4, self._last.confidence - 0.01 * self._misses),
                )
                return held

        if self.static_quad is not None:
            return frame_from_quad(self.static_quad, source="static", confidence=0.75)
        return identity_frame(width, height)
