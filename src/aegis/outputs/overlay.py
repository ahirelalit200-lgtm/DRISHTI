"""Annotated video overlay.

The overlay is not decoration: it is what gets recorded and streamed to ground,
so it has to carry enough state that someone watching only the video knows what
the system believed and when. It draws the rack frame, the calibrated zones, the
hand skeletons, the current instruction, and any active alert.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

try:  # pragma: no cover
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

POSE_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
]

COL_OK = (120, 220, 130)
COL_WARN = (60, 190, 250)
COL_CRIT = (70, 70, 245)
COL_INFO = (235, 190, 90)
COL_DIM = (120, 120, 120)
COL_TEXT = (240, 240, 240)


@dataclass
class OverlayState:
    instruction: str = ""
    step_label: str = ""
    progress: tuple[int, int] = (0, 0)
    alert: str = ""
    alert_severity: str = "info"
    alert_until: float = 0.0
    tier: str = ""
    fps: float = 0.0
    confidence: float = 0.0
    rack_source: str = ""
    blocked: bool = False
    session_id: str = ""
    recording: bool = False
    streaming: bool = False


def _put(img, text, org, scale=0.55, colour=COL_TEXT, thickness=1, shadow=True):
    if cv2 is None:
        return
    if shadow:
        cv2.putText(img, text, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def _panel(img, x, y, w, h, alpha=0.55, colour=(18, 14, 10)):
    if cv2 is None:
        return
    x, y = max(0, x), max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    if w <= 0 or h <= 0:
        return
    roi = img[y : y + h, x : x + w]
    overlay = np.full(roi.shape, colour, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def draw(frame: np.ndarray, result, rack, zones, state: OverlayState) -> np.ndarray:
    """Return an annotated copy of ``frame``."""
    if cv2 is None:
        return frame
    img = frame.copy()
    height, width = img.shape[:2]

    # --- rack quad ------------------------------------------------------
    if rack is not None and rack.corners_px is not None and rack.source != "identity":
        quad = rack.corners_px.astype(np.int32).reshape(-1, 1, 2)
        colour = COL_OK if rack.source == "aruco" else COL_INFO
        cv2.polylines(img, [quad], True, colour, 2, cv2.LINE_AA)
        _put(img, f"RACK FRAME [{rack.source}]", tuple(rack.corners_px[0].astype(int) + np.array([4, -8])), 0.45, colour)

    # --- zones ----------------------------------------------------------
    if zones is not None and rack is not None and len(zones):
        for zone in zones:
            try:
                pts = rack.to_pixels(zone.polygon).astype(np.int32).reshape(-1, 1, 2)
            except Exception:
                continue
            cv2.polylines(img, [pts], True, zone.colour, 1, cv2.LINE_AA)
            centre = pts.reshape(-1, 2).mean(axis=0).astype(int)
            _put(img, zone.label, (int(centre[0]) - 30, int(centre[1])), 0.42, zone.colour)

    # --- skeletons ------------------------------------------------------
    if result is not None:
        if result.pose is not None:
            pts = result.pose.points_px.astype(int)
            vis = result.pose.visibility
            for a, b in POSE_EDGES:
                if vis[a] > 0.4 and vis[b] > 0.4:
                    cv2.line(img, tuple(pts[a]), tuple(pts[b]), (150, 150, 165), 2, cv2.LINE_AA)
        for hand in result.hands:
            pts = hand.points_px.astype(int)
            colour = (255, 190, 100) if hand.label == "left" else (140, 230, 160)
            for a, b in HAND_EDGES:
                cv2.line(img, tuple(pts[a]), tuple(pts[b]), colour, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, tuple(p), 3, colour, -1, cv2.LINE_AA)
            wrist = pts[0]
            _put(img, f"{hand.label[:1].upper()} grip {hand.grip_aperture():.2f}", (wrist[0] - 20, wrist[1] + 22), 0.42, colour)

    # --- top status bar --------------------------------------------------
    _panel(img, 0, 0, width, 34, alpha=0.62, colour=(12, 16, 26))
    done, total = state.progress
    _put(img, "AEGIS  AI-HAR", (12, 23), 0.6, (245, 220, 150), 2)
    _put(img, f"STEP {done}/{total}", (170, 23), 0.55, COL_TEXT)
    _put(img, f"{state.tier}", (280, 23), 0.48, COL_INFO)
    _put(img, f"{state.fps:4.1f} fps", (width - 250, 23), 0.5, COL_DIM)
    if state.recording:
        cv2.circle(img, (width - 150, 17), 6, (60, 60, 240), -1, cv2.LINE_AA)
        _put(img, "REC", (width - 138, 23), 0.5, (90, 90, 245))
    if state.streaming:
        _put(img, "STREAM", (width - 90, 23), 0.5, (120, 220, 140))

    # --- progress strip --------------------------------------------------
    if total:
        bar_w = int(width * (done / total))
        cv2.rectangle(img, (0, 34), (bar_w, 38), (120, 220, 140), -1)

    # --- instruction panel ------------------------------------------------
    panel_h = 78
    _panel(img, 0, height - panel_h, width, panel_h, alpha=0.66, colour=(10, 14, 22))
    label_colour = COL_CRIT if state.blocked else (245, 220, 150)
    _put(img, state.step_label or "STANDBY", (14, height - panel_h + 24), 0.58, label_colour, 2)
    for i, line in enumerate(_wrap(state.instruction, 78)[:2]):
        _put(img, line, (14, height - panel_h + 48 + i * 22), 0.56, COL_TEXT)

    # --- alert banner -----------------------------------------------------
    if state.alert and time.monotonic() < state.alert_until:
        colour = {"critical": COL_CRIT, "warning": COL_WARN, "success": COL_OK}.get(state.alert_severity, COL_INFO)
        band_h = 44
        _panel(img, 0, 44, width, band_h, alpha=0.78, colour=(8, 8, 8))
        cv2.rectangle(img, (0, 44), (8, 44 + band_h), colour, -1)
        prefix = {"critical": "!! ", "warning": "! "}.get(state.alert_severity, "")
        _put(img, prefix + _wrap(state.alert, 84)[0], (20, 72), 0.62, colour, 2)

    if state.blocked:
        cv2.rectangle(img, (0, 0), (width - 1, height - 1), COL_CRIT, 6)

    # --- timestamp burn-in ------------------------------------------------
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    _put(img, f"{stamp}  {state.session_id}", (width - 340, height - 88), 0.45, (200, 200, 200))
    return img


def _wrap(text: str, width: int) -> list[str]:
    words = (text or "").split()
    if not words:
        return [""]
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def placeholder(width: int = 960, height: int = 540, message: str = "NO SIGNAL") -> np.ndarray:
    """A dark frame with a message, shown when the camera has not started."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (14, 12, 10)
    if cv2 is not None:
        for y in range(0, height, 28):
            cv2.line(img, (0, y), (width, y), (20, 18, 16), 1)
        _put(img, message, (width // 2 - 7 * len(message) // 2, height // 2), 0.85, (110, 130, 160), 2)
    return img
