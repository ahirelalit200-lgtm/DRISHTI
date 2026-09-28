"""Per-frame feature vector, expressed entirely in rack-relative terms.

Everything here is deliberately *relational* -- distances, angles and ratios
between the operator and the rack -- never an absolute image coordinate. Three
consequences that matter for this problem:

* **Orientation-agnostic.** Rotate the operator 180 degrees and the features are
  unchanged, because the rack defines the axes, not the floor.
* **Camera-agnostic.** Move the camera and re-calibrate the rack quad; the model
  keeps working without retraining.
* **Small.** ~90 floats per frame instead of a 1280x720 image, so a 32-frame
  window is 3k numbers. A GRU over that trains in minutes on a laptop GPU and
  runs at 200+ fps on CPU.

The vector layout is versioned and written into the model metadata, so a model
can never be loaded against a mismatched feature builder.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from aegis.perception.landmarks import PerceptionResult
from aegis.perception.zones import ZoneSet

FEATURE_VERSION = 2


@dataclass
class HandBlock:
    """The 13 numbers that describe one hand's state relative to the rack."""

    present: float = 0.0
    x: float = 0.5
    y: float = 0.5
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0
    grip: float = 0.0
    pinch: float = 0.0
    span: float = 0.0
    angle_sin: float = 0.0
    angle_cos: float = 0.0
    depth: float = 0.0
    stillness: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.present, self.x, self.y, self.vx, self.vy, self.speed,
                self.grip, self.pinch, self.span, self.angle_sin,
                self.angle_cos, self.depth, self.stillness,
            ],
            dtype=np.float32,
        )

    WIDTH = 13


class FeatureExtractor:
    """Turns a :class:`PerceptionResult` into a fixed-width float vector."""

    def __init__(self, zones: ZoneSet, *, history: int = 6, fps: float = 30.0) -> None:
        self.zones = zones
        self.fps = max(1.0, fps)
        self._prev: dict[str, np.ndarray] = {}
        self._trail: dict[str, deque] = {"left": deque(maxlen=history), "right": deque(maxlen=history)}

    @property
    def dimension(self) -> int:
        n_zones = len(self.zones)
        # two hand blocks + per-hand zone occupancy + pose block + interaction block
        return HandBlock.WIDTH * 2 + n_zones * 2 + 8 + 4

    def feature_names(self) -> list[str]:
        names: list[str] = []
        fields = [
            "present", "x", "y", "vx", "vy", "speed", "grip", "pinch",
            "span", "angle_sin", "angle_cos", "depth", "stillness",
        ]
        for side in ("left", "right"):
            names += [f"{side}_{f}" for f in fields]
        for side in ("left", "right"):
            names += [f"{side}_in_{z}" for z in self.zones.names]
        names += [
            "pose_present", "torso_sin", "torso_cos", "shoulder_span",
            "head_x", "head_y", "lean", "pose_visibility",
        ]
        names += ["hand_distance", "hands_together", "bimanual", "rack_confidence"]
        return names

    def reset(self) -> None:
        self._prev.clear()
        for trail in self._trail.values():
            trail.clear()

    # ------------------------------------------------------------------

    def _hand_block(self, result: PerceptionResult, side: str) -> tuple[HandBlock, np.ndarray]:
        hand = result.hand(side)
        zone_vec = np.zeros(len(self.zones), dtype=np.float32)
        if hand is None or hand.points_rack is None:
            self._prev.pop(side, None)
            self._trail[side].clear()
            return HandBlock(), zone_vec

        pts = np.asarray(hand.points_rack, dtype=np.float32)
        centre = pts.mean(axis=0)
        wrist = pts[0]

        prev = self._prev.get(side)
        if prev is None:
            vx = vy = 0.0
        else:
            vx = float((centre[0] - prev[0]) * self.fps)
            vy = float((centre[1] - prev[1]) * self.fps)
        self._prev[side] = centre.copy()
        self._trail[side].append(centre.copy())

        speed = float(math.hypot(vx, vy))
        stillness = 0.0
        if len(self._trail[side]) >= 3:
            arr = np.asarray(self._trail[side], dtype=np.float32)
            stillness = float(np.exp(-40.0 * arr.std(axis=0).mean()))

        # palm orientation in rack space -> survives an inverted operator
        axis = pts[9] - wrist
        angle = math.atan2(float(axis[1]), float(axis[0]))
        span = float(np.linalg.norm(pts[5] - pts[17]))

        depth = 0.0
        if hand.world is not None:
            depth = float(np.clip(hand.world[:, 2].mean() * 5.0 + 0.5, 0.0, 1.0))

        block = HandBlock(
            present=1.0,
            x=float(np.clip(centre[0], -1.0, 2.0)),
            y=float(np.clip(centre[1], -1.0, 2.0)),
            vx=float(np.clip(vx, -5.0, 5.0)),
            vy=float(np.clip(vy, -5.0, 5.0)),
            speed=float(np.clip(speed, 0.0, 5.0)),
            grip=float(np.clip(hand.grip_aperture(), 0.0, 5.0)),
            pinch=float(np.clip(hand.pinch_distance(), 0.0, 5.0)),
            span=float(np.clip(span, 0.0, 2.0)),
            angle_sin=math.sin(angle),
            angle_cos=math.cos(angle),
            depth=depth,
            stillness=stillness,
        )
        zone_vec = self.zones.occupancy_vector(centre)
        return block, zone_vec

    def _pose_block(self, result: PerceptionResult) -> np.ndarray:
        if result.pose is None or result.pose.points_rack is None:
            return np.zeros(8, dtype=np.float32)
        pts = np.asarray(result.pose.points_rack, dtype=np.float32)
        vis = result.pose.visibility
        shoulders = (pts[11] + pts[12]) / 2.0
        hips = (pts[23] + pts[24]) / 2.0
        axis = shoulders - hips
        angle = math.atan2(float(axis[1]), float(axis[0]))
        shoulder_span = float(np.linalg.norm(pts[11] - pts[12]))
        head = pts[0]
        lean = float(np.clip(abs(angle) / math.pi, 0.0, 1.0))
        return np.array(
            [
                1.0,
                math.sin(angle),
                math.cos(angle),
                float(np.clip(shoulder_span, 0.0, 2.0)),
                float(np.clip(head[0], -1.0, 2.0)),
                float(np.clip(head[1], -1.0, 2.0)),
                lean,
                float(np.clip(vis.mean(), 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def extract(self, result: PerceptionResult, rack_confidence: float = 0.5) -> np.ndarray:
        left, left_zones = self._hand_block(result, "left")
        right, right_zones = self._hand_block(result, "right")
        pose = self._pose_block(result)

        if left.present and right.present:
            hand_distance = float(math.hypot(left.x - right.x, left.y - right.y))
        else:
            hand_distance = 1.0
        together = float(np.exp(-10.0 * hand_distance))
        bimanual = 1.0 if (left.present and right.present) else 0.0

        vec = np.concatenate(
            [
                left.as_array(),
                right.as_array(),
                left_zones,
                right_zones,
                pose,
                np.array(
                    [
                        float(np.clip(hand_distance, 0.0, 3.0)),
                        together,
                        bimanual,
                        float(np.clip(rack_confidence, 0.0, 1.0)),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

        if vec.shape[0] != self.dimension:  # pragma: no cover - guard
            raise ValueError(f"feature width {vec.shape[0]} != declared {self.dimension}")
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


class WindowBuffer:
    """Fixed-length sliding window of feature vectors."""

    def __init__(self, length: int, dimension: int) -> None:
        self.length = length
        self.dimension = dimension
        self._buf: deque = deque(maxlen=length)

    def push(self, vec: np.ndarray) -> None:
        self._buf.append(np.asarray(vec, dtype=np.float32))

    def clear(self) -> None:
        self._buf.clear()

    @property
    def filled(self) -> bool:
        return len(self._buf) >= self.length

    @property
    def fill_ratio(self) -> float:
        return len(self._buf) / max(1, self.length)

    def array(self) -> np.ndarray:
        """(length, dimension), left-padded with the earliest frame if short."""
        if not self._buf:
            return np.zeros((self.length, self.dimension), dtype=np.float32)
        arr = np.asarray(self._buf, dtype=np.float32)
        if arr.shape[0] < self.length:
            pad = np.repeat(arr[:1], self.length - arr.shape[0], axis=0)
            arr = np.concatenate([pad, arr], axis=0)
        return arr
