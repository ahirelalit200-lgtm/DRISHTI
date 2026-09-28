"""Interaction zones defined in rack-normalised coordinates.

A zone is a named polygon over the payload rack -- 'storage_bay', 'main_slot',
'red_box', 'work_surface'. Because zones live in *rack* coordinates they are
recorded once and remain valid when the camera shifts or the operator rotates.

Zones do two jobs:

* They give the heuristic recogniser something concrete to reason about before
  any model is trained, so the system is useful on day one.
* They become input features for the learned model, which is what lets a small
  temporal network separate 'insert power cell' from 'insert data cable' even
  when the two hand motions look nearly identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass
class Zone:
    name: str
    polygon: np.ndarray            # (N, 2) in rack-normalised coords
    label: str = ""
    colour: tuple[int, int, int] = (90, 200, 255)

    def __post_init__(self) -> None:
        self.polygon = np.asarray(self.polygon, dtype=np.float32).reshape(-1, 2)
        if self.polygon.shape[0] < 3:
            raise ValueError(f"zone '{self.name}' needs at least 3 points")
        if not self.label:
            self.label = self.name.replace("_", " ").title()

    @property
    def centroid(self) -> np.ndarray:
        return self.polygon.mean(axis=0)

    def contains(self, point: Sequence[float]) -> bool:
        """Ray-casting point-in-polygon. Pure numpy: no OpenCV dependency."""
        x, y = float(point[0]), float(point[1])
        poly = self.polygon
        inside = False
        n = poly.shape[0]
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
                if x < x_cross:
                    inside = not inside
            j = i
        return inside

    def distance(self, point: Sequence[float]) -> float:
        """Distance from ``point`` to the polygon (0 when inside)."""
        if self.contains(point):
            return 0.0
        p = np.asarray(point, dtype=np.float32)
        poly = self.polygon
        best = float("inf")
        for i in range(poly.shape[0]):
            a = poly[i]
            b = poly[(i + 1) % poly.shape[0]]
            ab = b - a
            denom = float(ab @ ab) + 1e-12
            t = float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
            best = min(best, float(np.linalg.norm(p - (a + t * ab))))
        return best

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "polygon": [[round(float(x), 5), round(float(y), 5)] for x, y in self.polygon],
            "colour": list(self.colour),
        }


class ZoneSet:
    """An ordered collection of zones, persisted as JSON."""

    def __init__(self, zones: Iterable[Zone] | None = None) -> None:
        self.zones: list[Zone] = list(zones or [])

    def __len__(self) -> int:
        return len(self.zones)

    def __iter__(self):
        return iter(self.zones)

    @property
    def names(self) -> list[str]:
        return [z.name for z in self.zones]

    def get(self, name: str) -> Zone | None:
        for z in self.zones:
            if z.name == name:
                return z
        return None

    def locate(self, point_rack: Sequence[float]) -> str | None:
        """Name of the zone containing ``point_rack``, else None."""
        for z in self.zones:
            if z.contains(point_rack):
                return z.name
        return None

    def occupancy_vector(self, point_rack: Sequence[float] | None) -> np.ndarray:
        """Per-zone signal in [0, 1]: 1 inside, decaying with distance outside."""
        vec = np.zeros(len(self.zones), dtype=np.float32)
        if point_rack is None:
            return vec
        for i, z in enumerate(self.zones):
            d = z.distance(point_rack)
            vec[i] = float(np.exp(-6.0 * d))
        return vec

    # -------------------------------------------------------------- storage

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "zones": [z.to_dict() for z in self.zones]}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ZoneSet":
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
        zones = []
        for item in data.get("zones", []):
            try:
                zones.append(
                    Zone(
                        name=str(item["name"]),
                        polygon=np.asarray(item["polygon"], dtype=np.float32),
                        label=str(item.get("label", "")),
                        colour=tuple(item.get("colour", (90, 200, 255))),  # type: ignore
                    )
                )
            except Exception:
                continue
        return cls(zones)

    @classmethod
    def default_grid(cls, names: Sequence[str]) -> "ZoneSet":
        """Auto-generate placeholder zones so the app runs before calibration.

        Tiles the rack into a grid, one cell per protocol zone name. Crude on
        purpose -- the GUI flags it as uncalibrated so it is never mistaken for
        a real setup.
        """
        names = [n for n in names if n]
        if not names:
            return cls()
        cols = int(np.ceil(np.sqrt(len(names))))
        rows = int(np.ceil(len(names) / cols))
        palette = [
            (90, 200, 255), (255, 170, 80), (140, 230, 150),
            (235, 120, 200), (255, 230, 120), (150, 160, 255),
        ]
        zones = []
        pad = 0.03
        for i, name in enumerate(names):
            r, c = divmod(i, cols)
            x0, x1 = c / cols + pad, (c + 1) / cols - pad
            y0, y1 = r / rows + pad, (r + 1) / rows - pad
            zones.append(
                Zone(
                    name=name,
                    polygon=np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32),
                    colour=palette[i % len(palette)],
                )
            )
        return cls(zones)
