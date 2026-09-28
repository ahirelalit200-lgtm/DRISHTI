"""Video streaming to a specific IP.

Two modes, because "stream to a specific IP" means different things depending on
who initiates the connection:

* **Pull (default).** We host an MJPEG endpoint. The ground-station at the target
  IP opens ``http://<this-host>:<port>/stream.mjpg`` in a browser, VLC, or any
  HTTP client. Zero extra software at either end, and it degrades gracefully -
  if the link drops, the encoder keeps running and the client just reconnects.
* **Push (optional).** If ``stream_push_url`` is set and ffmpeg is on PATH, we
  additionally pipe frames into ffmpeg for RTSP/RTMP/UDP delivery to that URL.

Only the latest frame is ever held, so a slow client throttles itself instead of
back-pressuring the perception loop -- the same freshness-over-completeness
policy used in the capture thread.

Implemented on the standard library's ``http.server``: no FastAPI, no uvicorn,
one less thing to install on an offline machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import logging
import shutil
import socket
import subprocess
import threading
import time

import numpy as np

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


BOUNDARY = "aegisframe"


@dataclass
class StreamStatus:
    active: bool = False
    url: str = ""
    clients: int = 0
    frames_served: int = 0
    push_active: bool = False
    push_url: str = ""
    error: str = ""


class _LatestFrame:
    """Single-slot frame holder with a condition variable."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def wait(self, last_seq: int, timeout: float = 2.0) -> tuple[bytes | None, int]:
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._jpeg, self._seq


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>AEGIS HAR - Live Payload Feed</title>
<style>
 body{{margin:0;background:#05070d;color:#dbe6f5;font-family:Segoe UI,system-ui,sans-serif}}
 header{{padding:14px 20px;border-bottom:1px solid #1a2740;display:flex;gap:14px;align-items:center}}
 .dot{{width:10px;height:10px;border-radius:50%;background:#3ddc84;box-shadow:0 0 10px #3ddc84}}
 h1{{font-size:15px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;margin:0}}
 main{{display:flex;justify-content:center;padding:18px}}
 img{{max-width:100%;border:1px solid #1a2740;border-radius:10px}}
 footer{{padding:10px 20px;font-size:12px;color:#68809f}}
</style></head>
<body><header><span class="dot"></span><h1>AEGIS &mdash; On-board Experiment Feed</h1></header>
<main><img src="/stream.mjpg" alt="live feed"></main>
<footer>MJPEG over HTTP &middot; served locally from the payload computer</footer>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "AEGIS-HAR/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # silence per-request console spam
        return

    def do_GET(self) -> None:  # noqa: N802
        holder: _LatestFrame = self.server.frame_holder  # type: ignore[attr-defined]
        status: StreamStatus = self.server.stream_status  # type: ignore[attr-defined]

        if self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/healthz":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path not in ("/stream.mjpg", "/stream"):
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()
        status.clients += 1
        seq = -1
        try:
            while True:
                jpeg, seq = holder.wait(seq)
                if jpeg is None:
                    continue
                self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                status.frames_served += 1
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            status.clients = max(0, status.clients - 1)


class VideoStreamer:
    """Serves the annotated feed as MJPEG, optionally also pushing via ffmpeg."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8090,
        *,
        quality: int = 70,
        push_url: str = "",
        fps: int = 20,
    ) -> None:
        self.host = host
        self.port = port
        self.quality = int(np.clip(quality, 20, 95))
        self.push_url = push_url
        self.fps = max(1, fps)
        self._holder = _LatestFrame()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._status = StreamStatus()
        self._push: subprocess.Popen | None = None
        self._push_size: tuple[int, int] | None = None

    # ---------------------------------------------------------------- status

    def status(self) -> StreamStatus:
        return self._status

    @property
    def active(self) -> bool:
        return self._server is not None

    @staticmethod
    def local_ip() -> str:
        """Best-guess LAN address, so the GUI can print a reachable URL."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.2)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            try:
                return socket.gethostbyname(socket.gethostname())
            except Exception:
                return "127.0.0.1"

    # -------------------------------------------------------------- lifecycle

    def start(self) -> StreamStatus:
        if self._server is not None:
            return self._status
        try:
            server = ThreadingHTTPServer((self.host, self.port), _Handler)
            server.daemon_threads = True
            server.frame_holder = self._holder          # type: ignore[attr-defined]
            server.stream_status = self._status         # type: ignore[attr-defined]
            self._server = server
            self._thread = threading.Thread(target=server.serve_forever, name="mjpeg", daemon=True)
            self._thread.start()
            advertised = self.local_ip() if self.host in ("0.0.0.0", "") else self.host
            self._status.active = True
            self._status.url = f"http://{advertised}:{self.port}/"
            self._status.error = ""
            LOGGER.info("MJPEG stream live at %s", self._status.url)
        except OSError as exc:
            self._status.active = False
            self._status.error = f"port {self.port} unavailable: {exc}"
            LOGGER.error("stream start failed: %s", exc)
        return self._status

    def _start_push(self, width: int, height: int) -> None:
        if not self.push_url or self._push is not None:
            return
        if shutil.which("ffmpeg") is None:
            self._status.error = "ffmpeg not found on PATH - push disabled"
            return
        protocol = "rtsp" if self.push_url.startswith("rtsp") else "flv" if self.push_url.startswith("rtmp") else "mpegts"
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p", "-f", protocol, self.push_url,
        ]
        try:
            self._push = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._push_size = (width, height)
            self._status.push_active = True
            self._status.push_url = self.push_url
            LOGGER.info("ffmpeg push started -> %s", self.push_url)
        except Exception as exc:  # pragma: no cover
            self._status.error = f"push failed: {exc}"
            self._push = None

    def publish(self, image: np.ndarray) -> None:
        if cv2 is None or self._server is None:
            return
        try:
            ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            if ok:
                self._holder.publish(buf.tobytes())
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("encode failed: %s", exc)

        if self.push_url:
            height, width = image.shape[:2]
            if self._push is None:
                self._start_push(width, height)
            if self._push is not None and self._push.stdin is not None:
                try:
                    frame = image
                    if self._push_size and (width, height) != self._push_size:
                        frame = cv2.resize(image, self._push_size)
                    self._push.stdin.write(frame.tobytes())
                except Exception:
                    self._status.push_active = False
                    try:
                        self._push.kill()
                    except Exception:
                        pass
                    self._push = None

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
        self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._push is not None:
            try:
                if self._push.stdin:
                    self._push.stdin.close()
                self._push.terminate()
            except Exception:
                pass
            self._push = None
        self._status.active = False
        self._status.push_active = False
        self._status.clients = 0
