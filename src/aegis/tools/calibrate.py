"""Interactive calibration: define the rack quad and the interaction zones.

Run once per physical setup. Two stages, both point-and-click on a live frame:

1. **Rack quad** -- click the four corners of the payload rack, clockwise from
   top-left. This is the static fallback used whenever ArUco markers are hidden
   by the operator's arms. Skippable if you are using markers exclusively.
2. **Zones** -- for each zone named in ``configs/protocol.yaml``, click a polygon
   around the corresponding physical region.

Zones are stored in *rack-normalised* coordinates, so they stay correct if the
camera shifts, if the rack rotates, or if the operator is inverted.

Controls
    left click    add a point
    right click   undo the last point
    ENTER / n     accept the current polygon, move to the next zone
    s             skip this zone
    r             restart the current polygon
    f             freeze / unfreeze the video (freeze before clicking)
    q / ESC       save what has been defined and quit
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np

from aegis.capture.source import VideoSource
from aegis.config import load_config
from aegis.perception.rack_frame import RackTracker, frame_from_quad, identity_frame, order_quad
from aegis.perception.zones import Zone, ZoneSet
from aegis.protocol.spec import load_protocol

LOGGER = logging.getLogger(__name__)

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


PALETTE = [
    (90, 200, 255), (255, 170, 80), (140, 230, 150),
    (235, 120, 200), (255, 230, 120), (150, 160, 255),
    (120, 255, 220), (200, 200, 200),
]


class ClickCollector:
    """Accumulates polygon points from mouse events."""

    def __init__(self) -> None:
        self.points: list[tuple[int, int]] = []
        self.hover: tuple[int, int] = (0, 0)

    def callback(self, event, x, y, flags, param) -> None:  # pragma: no cover - UI
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def reset(self) -> None:
        self.points.clear()


def _banner(img, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    height, width = img.shape[:2]
    box = 26 * len(lines) + 16
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (width, box), (10, 12, 18), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(img, text, (14, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (14, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)


def _draw_polygon(img, points, colour, closed: bool = False, hover=None) -> None:
    for i, p in enumerate(points):
        cv2.circle(img, p, 5, colour, -1, cv2.LINE_AA)
        cv2.putText(img, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    if len(points) > 1:
        cv2.polylines(img, [np.array(points, np.int32)], closed, colour, 2, cv2.LINE_AA)
    if hover is not None and points and not closed:
        cv2.line(img, points[-1], hover, colour, 1, cv2.LINE_AA)


def calibrate(config_path: str | None = None, source_override=None) -> int:
    if cv2 is None:
        print("ERROR: OpenCV is not installed. Run INSTALL.bat first.")
        return 2

    config = load_config(config_path)
    protocol = load_protocol(config.protocol_file)
    zone_names = list(protocol.zone_names)
    if not zone_names:
        print("This protocol declares no zones. Nothing to calibrate.")
        print("Add a 'zone:' field to steps in configs/protocol.yaml if you want zone gating.")
        return 0

    source = VideoSource(
        source_override if source_override is not None else config.resolved_source(),
        width=config.frame_width,
        height=config.frame_height,
        fps=config.target_fps,
        flip_horizontal=config.flip_horizontal,
        loop_file=True,
    )
    if not source.start():
        print(f"ERROR: {source.error}")
        return 2

    window = "AEGIS Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1180, 700)
    collector = ClickCollector()
    cv2.setMouseCallback(window, collector.callback)

    rack_tracker = RackTracker(
        enabled=config.rack_marker_enabled,
        dictionary=config.rack_marker_dict,
        marker_ids=config.rack_marker_ids,
    )

    frozen: np.ndarray | None = None
    stage = "rack"
    zone_index = 0
    saved_zones: list[Zone] = []
    rack_quad: np.ndarray | None = None
    existing_quad_path = config.zones_file.with_name("rack_quad.json")

    print(__doc__)

    while True:
        live = source.read(timeout=0.1)
        if live is not None and frozen is None:
            display_source = live.image
        elif frozen is not None:
            display_source = frozen
        else:
            continue

        img = display_source.copy()
        rack = rack_tracker.update(display_source)
        if rack_quad is not None:
            rack = frame_from_quad(rack_quad, source="static", confidence=0.8)

        if rack.corners_px is not None and rack.source != "identity":
            cv2.polylines(img, [rack.corners_px.astype(np.int32).reshape(-1, 1, 2)], True,
                          (120, 220, 140), 2, cv2.LINE_AA)

        for i, zone in enumerate(saved_zones):
            try:
                pts = rack.to_pixels(zone.polygon).astype(np.int32)
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, zone.colour, 2, cv2.LINE_AA)
                centre = pts.mean(axis=0).astype(int)
                cv2.putText(img, zone.name, (int(centre[0]) - 40, int(centre[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, zone.colour, 2, cv2.LINE_AA)
            except Exception:
                pass

        if stage == "rack":
            colour = (120, 220, 140)
            _draw_polygon(img, collector.points, colour, closed=len(collector.points) >= 4,
                          hover=collector.hover)
            _banner(img, [
                ("STAGE 1/2  -  RACK QUAD", (240, 220, 150)),
                (f"Click the 4 corners of the payload rack, clockwise from TOP-LEFT  ({len(collector.points)}/4)",
                 (220, 230, 240)),
                ("ENTER accept   R restart   F freeze   S skip (use ArUco only)   Q quit",
                 (140, 165, 195)),
            ])
        else:
            name = zone_names[zone_index]
            colour = PALETTE[zone_index % len(PALETTE)]
            _draw_polygon(img, collector.points, colour, closed=False, hover=collector.hover)
            _banner(img, [
                (f"STAGE 2/2  -  ZONE {zone_index + 1}/{len(zone_names)}:  {name.upper()}", (240, 220, 150)),
                (f"Click a polygon around the '{name}' region   ({len(collector.points)} points, need 3+)",
                 (220, 230, 240)),
                ("ENTER next   R restart   S skip   F freeze   Q save+quit", (140, 165, 195)),
            ])

        if frozen is not None:
            cv2.putText(img, "FROZEN", (img.shape[1] - 130, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (90, 200, 255), 2, cv2.LINE_AA)

        cv2.imshow(window, img)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("f"):
            frozen = None if frozen is not None else display_source.copy()
            continue
        if key == ord("r"):
            collector.reset()
            continue

        if stage == "rack":
            if key == ord("s"):
                stage = "zones"
                collector.reset()
                print("Rack quad skipped - relying on ArUco markers only.")
                continue
            if key in (13, 10, ord("n")) and len(collector.points) >= 4:
                rack_quad = order_quad(np.array(collector.points[:4], dtype=np.float32))
                existing_quad_path.parent.mkdir(parents=True, exist_ok=True)
                existing_quad_path.write_text(
                    json.dumps({"quad": rack_quad.tolist(),
                                "frame_size": [display_source.shape[1], display_source.shape[0]]}, indent=2),
                    encoding="utf-8",
                )
                print(f"Rack quad saved -> {existing_quad_path}")
                stage = "zones"
                collector.reset()
            continue

        # ---- zone stage ----
        if key == ord("s"):
            print(f"Skipped zone '{zone_names[zone_index]}'")
            collector.reset()
            zone_index += 1
            if zone_index >= len(zone_names):
                break
            continue

        if key in (13, 10, ord("n")):
            if len(collector.points) < 3:
                print("Need at least 3 points before accepting.")
                continue
            pts_px = np.array(collector.points, dtype=np.float32)
            pts_rack = rack.to_rack(pts_px)
            name = zone_names[zone_index]
            saved_zones = [z for z in saved_zones if z.name != name]
            saved_zones.append(Zone(name=name, polygon=pts_rack,
                                    colour=PALETTE[zone_index % len(PALETTE)]))
            print(f"Zone '{name}' captured with {len(collector.points)} points.")
            collector.reset()
            zone_index += 1
            if zone_index >= len(zone_names):
                break

    source.stop()
    cv2.destroyAllWindows()

    if saved_zones:
        zone_set = ZoneSet(saved_zones)
        path = zone_set.save(config.zones_file)
        print()
        print(f"Saved {len(saved_zones)} zone(s) -> {path}")
        missing = [n for n in zone_names if n not in zone_set.names]
        if missing:
            print(f"WARNING: no polygon captured for: {', '.join(missing)}")
            print("Those steps will fall back to a lower-confidence 'anywhere' match.")
        return 0

    print("No zones were captured. configs/zones.json was left unchanged.")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate rack frame and interaction zones")
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", default=None, help="override video source")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    source = args.source
    if source is not None and str(source).isdigit():
        source = int(source)
    try:
        return calibrate(args.config, source)
    except KeyboardInterrupt:
        print("\nCalibration cancelled.")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
