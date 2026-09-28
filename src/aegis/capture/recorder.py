"""Local video storage.

The statement asks for the experiment video to be stored locally *as well as*
streamed. Writing happens on a dedicated thread with a bounded queue: if the
disk stalls, we drop frames from the recording rather than stall the perception
loop, because a missing frame in the archive is far cheaper than a late warning
to the operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import queue
import threading
import time

import numpy as np

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


@dataclass
class RecorderStatus:
    active: bool = False
    path: str = ""
    frames: int = 0
    dropped: int = 0
    seconds: float = 0.0
    error: str = ""
    size_mb: float = 0.0


class VideoRecorder:
    def __init__(self, directory: str | Path, fps: int = 20, *, queue_size: int = 90) -> None:
        self.directory = Path(directory)
        self.fps = max(1, int(fps))
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._writer = None
        self._lock = threading.Lock()
        self._status = RecorderStatus()
        self._started_at = 0.0
        self._size = (0, 0)

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> RecorderStatus:
        with self._lock:
            s = RecorderStatus(**vars(self._status))
        if s.active:
            s.seconds = max(0.0, time.monotonic() - self._started_at)
            try:
                if s.path:
                    s.size_mb = round(Path(s.path).stat().st_size / 1e6, 2)
            except Exception:
                pass
        return s

    def start(self, session_id: str, frame_shape: tuple[int, int]) -> Path | None:
        """Open a writer sized for ``frame_shape`` = (height, width)."""
        if cv2 is None:
            with self._lock:
                self._status.error = "OpenCV not installed"
            return None
        if self.active:
            return Path(self._status.path)

        self.directory.mkdir(parents=True, exist_ok=True)
        height, width = int(frame_shape[0]), int(frame_shape[1])
        self._size = (width, height)
        path = self.directory / f"{session_id}.mp4"

        writer = None
        for codec in ("mp4v", "avc1", "MJPG"):
            candidate_path = path if codec != "MJPG" else path.with_suffix(".avi")
            fourcc = cv2.VideoWriter_fourcc(*codec)
            candidate = cv2.VideoWriter(str(candidate_path), fourcc, float(self.fps), (width, height))
            if candidate.isOpened():
                writer, path = candidate, candidate_path
                break
            candidate.release()
        if writer is None:
            with self._lock:
                self._status.error = "no usable video codec (tried mp4v, avc1, MJPG)"
            return None

        self._writer = writer
        self._stop.clear()
        self._started_at = time.monotonic()
        with self._lock:
            self._status = RecorderStatus(active=True, path=str(path))
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()
        LOGGER.info("recording to %s", path)
        return path

    def write(self, image: np.ndarray) -> None:
        if not self.active:
            return
        try:
            self._queue.put_nowait(image)
        except queue.Full:
            with self._lock:
                self._status.dropped += 1

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                image = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                if cv2 is not None and (image.shape[1], image.shape[0]) != self._size:
                    image = cv2.resize(image, self._size)
                self._writer.write(image)
                with self._lock:
                    self._status.frames += 1
            except Exception as exc:  # pragma: no cover
                with self._lock:
                    self._status.error = str(exc)
                break

    def stop(self) -> RecorderStatus:
        final = self.status()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=4.0)
        self._thread = None
        try:
            if self._writer is not None:
                self._writer.release()
        except Exception:
            pass
        self._writer = None
        with self._lock:
            self._status.active = False
            final.active = False
            final.frames = self._status.frames
        LOGGER.info("recording stopped: %s frames", final.frames)
        return final


def new_session_id(prefix: str = "session") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
