"""Rack-frame tests -- the orientation-agnostic claim, verified.

The system's central claim is that recognition does not depend on which way up
the operator (or the rack) happens to be. That is only true if the rack frame is
recovered correctly at arbitrary rotations, so these tests synthesise a scene
with ArUco markers, rotate it, and check that a fixed physical point keeps the
same rack-relative coordinates.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

cv2 = pytest.importorskip("cv2")
if not hasattr(cv2, "aruco"):
    pytest.skip("cv2.aruco unavailable (needs opencv-contrib-python)", allow_module_level=True)

from aegis.perception.rack_frame import (
    RackTracker,
    frame_from_quad,
    identity_frame,
    order_quad,
)
from aegis.perception.zones import Zone, ZoneSet


MARKER_PX = 70


def render_scene(size=(900, 700), centre=None, rotation_deg=0.0, span=260, ids=(0, 1, 2, 3)):
    """Draw four ArUco markers arranged as a rotated square around ``centre``."""
    width, height = size
    img = np.full((height, width, 3), 235, np.uint8)
    centre = centre or (width / 2, height / 2)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    theta = math.radians(rotation_deg)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    offsets = np.array([[-span, -span], [span, -span], [span, span], [-span, span]], dtype=np.float32)
    placed = []
    for marker_id, offset in zip(ids, offsets):
        pos = rot @ offset + np.asarray(centre)
        marker = cv2.aruco.generateImageMarker(aruco_dict, int(marker_id), MARKER_PX)
        marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        if rotation_deg:
            m = cv2.getRotationMatrix2D((MARKER_PX / 2, MARKER_PX / 2), -rotation_deg, 1.0)
            marker = cv2.warpAffine(marker, m, (MARKER_PX, MARKER_PX), borderValue=(235, 235, 235))
        x0 = int(pos[0] - MARKER_PX / 2)
        y0 = int(pos[1] - MARKER_PX / 2)
        if x0 < 0 or y0 < 0 or x0 + MARKER_PX > width or y0 + MARKER_PX > height:
            return None, None
        img[y0 : y0 + MARKER_PX, x0 : x0 + MARKER_PX] = marker
        placed.append(pos)
    return img, np.asarray(placed, dtype=np.float32)


def test_tracker_detects_markers_and_builds_a_frame():
    img, _ = render_scene()
    tracker = RackTracker(enabled=True)
    rack = tracker.update(img)
    assert rack.source == "aruco", "markers should have been detected"
    assert rack.valid
    assert rack.confidence > 0.8


def test_rack_coordinates_are_invariant_to_rack_rotation():
    """A point at the rack's centre must map to (0.5, 0.5) at any rotation."""
    results = []
    for angle in (0.0, 30.0, 90.0, 180.0, 270.0):
        img, corners = render_scene(rotation_deg=angle)
        if img is None:
            continue
        tracker = RackTracker(enabled=True)
        rack = tracker.update(img)
        assert rack.source == "aruco", f"markers not found at {angle} deg"
        centre_px = corners.mean(axis=0)
        centre_rack = rack.to_rack(centre_px.reshape(1, 2))[0]
        results.append((angle, centre_rack))

    assert len(results) >= 4, "not enough rotations rendered"
    for angle, centre_rack in results:
        assert abs(centre_rack[0] - 0.5) < 0.06, f"x drifted at {angle} deg: {centre_rack}"
        assert abs(centre_rack[1] - 0.5) < 0.06, f"y drifted at {angle} deg: {centre_rack}"


def test_rack_coordinates_are_invariant_to_rack_translation():
    """Moving the whole rig must not change rack-relative coordinates."""
    seen = []
    for centre in ((450, 350), (380, 300), (520, 400)):
        img, corners = render_scene(centre=centre)
        if img is None:
            continue
        rack = RackTracker(enabled=True).update(img)
        assert rack.source == "aruco"
        probe_px = corners[0] + (corners[2] - corners[0]) * 0.25
        seen.append(rack.to_rack(probe_px.reshape(1, 2))[0])
    assert len(seen) >= 2
    reference = seen[0]
    for point in seen[1:]:
        assert np.allclose(point, reference, atol=0.06), f"{point} != {reference}"


def test_occlusion_holds_the_last_good_frame():
    """Losing the markers briefly must not collapse the frame to identity."""
    img, _ = render_scene()
    tracker = RackTracker(enabled=True, grace_frames=10)
    good = tracker.update(img)
    assert good.source == "aruco"

    blank = np.full_like(img, 235)
    held = tracker.update(blank)
    assert held.source == "aruco_hold", "should hold the previous frame during occlusion"
    assert np.allclose(held.homography, good.homography)

    for _ in range(12):
        final = tracker.update(blank)
    assert final.source == "identity", "grace period must eventually expire"


def test_static_quad_used_when_markers_absent():
    quad = [[100, 80], [500, 80], [500, 400], [100, 400]]
    tracker = RackTracker(enabled=False, static_quad=quad)
    rack = tracker.update(np.zeros((480, 640, 3), np.uint8))
    assert rack.source == "static"
    centre = rack.to_rack(np.array([[300, 240]], np.float32))[0]
    assert abs(centre[0] - 0.5) < 0.02
    assert abs(centre[1] - 0.5) < 0.02


def test_round_trip_pixels_to_rack_and_back():
    rack = frame_from_quad([[10, 20], [610, 30], [600, 460], [20, 450]])
    points = np.array([[100, 100], [300, 250], [580, 430]], np.float32)
    recovered = rack.to_pixels(rack.to_rack(points))
    assert np.allclose(points, recovered, atol=1e-2)


def test_order_quad_normalises_corner_order():
    ordered = order_quad(np.array([[500, 400], [100, 80], [500, 80], [100, 400]], np.float32))
    assert ordered.shape == (4, 2)
    assert tuple(ordered[0]) == (100.0, 80.0), "top-left must come first"


def test_zone_membership_in_rack_space():
    zones = ZoneSet([
        Zone("left_bay", np.array([[0.0, 0.0], [0.45, 0.0], [0.45, 1.0], [0.0, 1.0]], np.float32)),
        Zone("right_bay", np.array([[0.55, 0.0], [1.0, 0.0], [1.0, 1.0], [0.55, 1.0]], np.float32)),
    ])
    assert zones.locate((0.2, 0.5)) == "left_bay"
    assert zones.locate((0.8, 0.5)) == "right_bay"
    assert zones.locate((0.5, 0.5)) is None, "the gap belongs to neither zone"

    occupancy = zones.occupancy_vector((0.2, 0.5))
    assert occupancy[0] == pytest.approx(1.0)
    assert occupancy[1] < 1.0

    assert zones.get("left_bay").distance((0.2, 0.5)) == 0.0
    assert zones.get("left_bay").distance((0.65, 0.5)) == pytest.approx(0.2, abs=1e-3)


def test_identity_frame_normalises_by_image_size():
    rack = identity_frame(640, 480)
    point = rack.to_rack(np.array([[320, 240]], np.float32))[0]
    assert point == pytest.approx([0.5, 0.5], abs=1e-3)
