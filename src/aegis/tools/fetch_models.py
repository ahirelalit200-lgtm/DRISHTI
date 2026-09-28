"""Fetch the MediaPipe ``.task`` bundles needed by the Tasks backend.

Only required if your MediaPipe wheel has no ``mp.solutions`` (newer/slim
wheels). Run once on a machine with internet; after that the app is fully
offline, which is what the deliverable requires.

The files are small (~13 MB total) and are committed to the ``models/mediapipe``
folder, so a packaged build carries them and needs no network at all.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import urllib.error
import urllib.request

BASE = "https://storage.googleapis.com/mediapipe-models"

ASSETS = {
    "hand_landmarker.task": f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

# Optional heavier pose model: better joints, ~3x the cost.
OPTIONAL = {
    "pose_landmarker_full.task": f"{BASE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
}


def _progress(name: str):
    def hook(block_num, block_size, total_size):
        if total_size <= 0:
            return
        done = min(total_size, block_num * block_size)
        pct = done * 100 / total_size
        bar = "#" * int(pct / 3.3)
        sys.stdout.write(f"\r  {name:<32} [{bar:<30}] {pct:5.1f}%")
        sys.stdout.flush()

    return hook


def fetch(name: str, url: str, target_dir: Path, force: bool = False) -> bool:
    target = target_dir / name
    if target.exists() and not force:
        size = target.stat().st_size / 1e6
        print(f"  {name:<32} already present ({size:.1f} MB)")
        return True
    try:
        urllib.request.urlretrieve(url, target, _progress(name))
        print()
        size = target.stat().st_size / 1e6
        digest = hashlib.sha256(target.read_bytes()).hexdigest()[:16]
        print(f"  {name:<32} downloaded {size:.1f} MB  sha256:{digest}")
        return True
    except urllib.error.URLError as exc:
        print()
        print(f"  {name:<32} FAILED: {exc}")
        if target.exists():
            target.unlink()
        return False
    except Exception as exc:
        print()
        print(f"  {name:<32} FAILED: {exc}")
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download MediaPipe task bundles")
    parser.add_argument("--dir", default=None, help="target directory")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--full-pose", action="store_true", help="also fetch the heavier pose model")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[3]
    target_dir = Path(args.dir) if args.dir else root / "models" / "mediapipe"
    target_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  Fetching MediaPipe model bundles")
    print(f"  -> {target_dir}")
    print("=" * 62)

    assets = dict(ASSETS)
    if args.full_pose:
        assets.update(OPTIONAL)

    results = [fetch(name, url, target_dir, args.force) for name, url in assets.items()]

    print()
    if all(results):
        print("All model bundles are in place. The app can now run fully offline.")
        return 0

    print("Some downloads failed.")
    print()
    print("If this machine has no internet, download these on another machine")
    print(f"and copy them into {target_dir}:")
    for name, url in assets.items():
        if not (target_dir / name).exists():
            print(f"  {url}")
    print()
    print("Alternatively, install a MediaPipe wheel that includes the classic")
    print("solutions API, which needs no separate model files:")
    print("  pip install mediapipe==0.10.14")
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
