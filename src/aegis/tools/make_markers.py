"""Generate a printable ArUco marker sheet for the payload rack.

Tape one marker near each corner of the rack, in the order 0 (top-left),
1 (top-right), 2 (bottom-right), 3 (bottom-left). The app then recovers the
rack's pose every frame, which is what makes the recognition orientation-
agnostic: an operator working upside-down produces identical rack-relative
features to one working upright.

Print at 100% scale (no "fit to page"). Bigger markers detect from further
away; 60-80 mm works well for a desk-sized rig at 1-2 m.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


CORNER_NAMES = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]


def build_sheet(dictionary: str, ids: list[int], px: int, dpi: int) -> np.ndarray:
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    quiet = px // 5
    cell = px + quiet * 2
    label_h = 46

    cols, rows = 2, 2
    sheet_w = cols * cell + 60
    sheet_h = rows * (cell + label_h) + 130
    sheet = np.full((sheet_h, sheet_w, 3), 255, np.uint8)

    cv2.putText(sheet, "AEGIS  -  Payload Rack Fiducials", (30, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(sheet, f"{dictionary}   {px}px @ {dpi}dpi = {px / dpi * 25.4:.0f}mm   Print at 100% scale",
                (30, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1, cv2.LINE_AA)

    for i, marker_id in enumerate(ids[:4]):
        r, c = divmod(i, cols)
        x = 30 + c * cell
        y = 100 + r * (cell + label_h)
        img = cv2.aruco.generateImageMarker(aruco_dict, int(marker_id), px)
        tile = np.full((cell, cell), 255, np.uint8)
        tile[quiet : quiet + px, quiet : quiet + px] = img
        sheet[y : y + cell, x : x + cell] = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(sheet, (x, y), (x + cell, y + cell), (200, 200, 200), 1)
        cv2.putText(sheet, f"ID {marker_id}  -  {CORNER_NAMES[i]}", (x + 4, y + cell + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    footer = sheet_h - 26
    cv2.putText(sheet, "Cut out, then tape each marker flat at the matching corner of the rack.",
                (30, footer), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 90, 90), 1, cv2.LINE_AA)
    return sheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate printable ArUco rack markers")
    parser.add_argument("--dict", default="DICT_4X4_50", help="ArUco dictionary name")
    parser.add_argument("--ids", default="0,1,2,3", help="four marker ids, clockwise from top-left")
    parser.add_argument("--size", type=int, default=600, help="marker size in pixels")
    parser.add_argument("--dpi", type=int, default=200, help="assumed print DPI, for the mm hint")
    parser.add_argument("--out", default="docs/rack_markers.png")
    args = parser.parse_args(argv)

    if cv2 is None or not hasattr(cv2, "aruco"):
        print("ERROR: cv2.aruco unavailable. Install opencv-contrib-python:")
        print("  .venv\\Scripts\\python.exe -m pip install opencv-contrib-python")
        return 2

    try:
        ids = [int(v) for v in args.ids.split(",")]
    except ValueError:
        print("ERROR: --ids must be comma-separated integers, e.g. 0,1,2,3")
        return 2
    if len(ids) < 4:
        print("ERROR: four marker ids are required.")
        return 2

    root = Path(__file__).resolve().parents[3]
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    sheet = build_sheet(args.dict, ids, args.size, args.dpi)
    cv2.imwrite(str(out), sheet)

    print("=" * 62)
    print(f"Marker sheet written to: {out}")
    print(f"Dictionary: {args.dict}   IDs: {ids[:4]}")
    print(f"Each marker prints at about {args.size / args.dpi * 25.4:.0f} mm at {args.dpi} dpi.")
    print("=" * 62)
    print()
    print("1. Print at 100% scale (turn OFF 'fit to page').")
    print("2. Tape ID 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left")
    print("   on the payload rack, flat and fully visible to the camera.")
    print("3. Run CALIBRATE_ZONES.bat -- the green quad confirms detection.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
