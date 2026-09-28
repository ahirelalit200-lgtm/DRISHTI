"""Threaded video ingestion with drop-oldest semantics.

Reading frames on the inference thread is the classic way to end up with a
growing latency backlog: the camera keeps buffering, you keep processing
increasingly stale frames, and by the end of a five-minute run the overlay is
several seconds behind reality. For a system whose whole job is to warn an
operator *before* they make the next mistake, that is a correctness bug, not a
performance one.

So the grabber runs in its own thread and keeps exactly one frame. Slow
inference costs you frames, never freshness.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


@dataclass
class Frame:
    index: int
    image: np.ndarray
    timestamp: float          # time.time() at capture
    monotonic: float          # time.monotonic() at capture


class VideoSource:
    """Opens a webcam index, video file, or RTSP/HTTP URL in a worker thread."""

    def __init__(
        self,
        source: Any = 0,
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        flip_horizontal: bool = True,
        loop_file: bool = True,
        reconnect_delay_s: float = 2.0,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_horizontal = flip_horizontal
        self.loop_file = loop_file
        self.reconnect_delay_s = reconnect_delay_s

        self._cap: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._index = 0
        self._dropped = 0
        self._opened = False
        self._error = ""
        self._is_file = not isinstance(source, int) and "://" not in str(source)
        self._grab_times: list[float] = []

    # ------------------------------------------------------------- lifecycle

    @property
    def opened(self) -> bool:
        return self._opened

    @property
    def error(self) -> str:
        return self._error

    @property
    def dropped(self) -> int:
        return self._dropped

    def describe(self) -> str:
        if isinstance(self.source, int):
            return f"Webcam #{self.source}"
        text = str(self.source)
        return text if len(text) < 48 else "..." + text[-45:]

    def _open(self) -> bool:
        if cv2 is None:
            self._error = "OpenCV is not installed - run INSTALL.bat"
            return False
        try:
            if isinstance(self.source, int):
                backends = []
                if hasattr(cv2, "CAP_DSHOW"):
                    backends.append(cv2.CAP_DSHOW)   # Windows: fastest to open
                if hasattr(cv2, "CAP_MSMF"):
                    backends.append(cv2.CAP_MSMF)
                backends.append(cv2.CAP_ANY)
                cap = None
                for backend in backends:
                    candidate = cv2.VideoCapture(self.source, backend)
                    if candidate.isOpened():
                        cap = candidate
                        break
                    candidate.release()
                if cap is None:
                    self._error = (
                        f"Could not open camera #{self.source}. "
                        "Close other apps using the webcam, or set video_source in configs/app.yaml."
                    )
                    return False
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
            else:
                cap = cv2.VideoCapture(str(self.source))
                if not cap.isOpened():
                    self._error = f"Could not open source: {self.source}"
                    return False
            self._cap = cap
            self._error = ""
            return True
        except Exception as exc:  # pragma: no cover
            self._error = f"{type(exc).__name__}: {exc}"
            return False

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return self._opened
        self._stop.clear()
        if not self._open():
            self._opened = False
            return False
        self._opened = True
        self._thread = threading.Thread(target=self._run, name="video-source", daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._cap is None:
                if not self._open():
                    time.sleep(self.reconnect_delay_s)
                    continue
            ok, image = self._cap.read()
            if not ok or image is None:
                if self._is_file and self.loop_file:
                    try:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    except Exception:
                        pass
                LOGGER.warning("frame grab failed; reconnecting")
                self._release_capture()
                self._opened = False
                time.sleep(self.reconnect_delay_s)
                continue

            if self.flip_horizontal and cv2 is not None:
                image = cv2.flip(image, 1)

            now_wall, now_mono = time.time(), time.monotonic()
            self._index += 1
            frame = Frame(self._index, image, now_wall, now_mono)
            with self._lock:
                if self._latest is not None:
                    self._dropped += 1
                self._latest = frame
                self._opened = True
                self._grab_times.append(now_mono)
                if len(self._grab_times) > 60:
                    self._grab_times = self._grab_times[-60:]

            if self._is_file:
                time.sleep(max(0.0, 1.0 / max(1, self.fps)))

    def read(self, *, timeout: float = 0.0) -> Frame | None:
        """Take the newest frame, consuming it. None if nothing new."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                frame, self._latest = self._latest, None
            if frame is not None or timeout <= 0 or time.monotonic() >= deadline:
                return frame
            time.sleep(0.002)

    def peek(self) -> Frame | None:
        with self._lock:
            return self._latest

    def measured_fps(self) -> float:
        with self._lock:
            times = list(self._grab_times)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 1e-6 else 0.0

    def _release_capture(self) -> None:
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._release_capture()
        self._opened = False


def probe_cameras(max_index: int = 5) -> list[int]:
    """Return webcam indices that can actually be opened. Used by diagnostics."""
    found: list[int] = []
    if cv2 is None:
        return found
    for i in range(max_index):
        cap = None
        try:
            backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(i)
        except Exception:
            pass
        finally:
            if cap is not None:
                cap.release()
    return found
