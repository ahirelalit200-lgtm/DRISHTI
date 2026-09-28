"""Typed application configuration loaded from ``configs/app.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Repository root, resolved from this file's location (src/aegis/config.py)."""
    return Path(__file__).resolve().parents[2]


def _resolve(path_like: Any, root: Path) -> Path:
    p = Path(str(path_like)).expanduser()
    return p if p.is_absolute() else (root / p)


@dataclass
class AppConfig:
    # --- capture -------------------------------------------------------
    video_source: Any = 0                 # int webcam index, path, or rtsp:// url
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 30
    flip_horizontal: bool = True          # mirror webcam so left/right read naturally
    loop_video_file: bool = True

    # --- perception ----------------------------------------------------
    pose_enabled: bool = True
    hands_enabled: bool = True
    max_hands: int = 2
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    model_complexity: int = 1

    rack_marker_enabled: bool = True      # ArUco-based rack-relative frame
    rack_marker_dict: str = "DICT_4X4_50"
    rack_marker_ids: list[int] = field(default_factory=lambda: [0, 1, 2, 3])

    objects_enabled: bool = False         # optional custom YOLO detector
    objects_model_path: str = "models/objects/best.pt"
    objects_confidence: float = 0.45

    # --- recognition ---------------------------------------------------
    action_model_path: str = "models/action/action_model.onnx"
    action_metadata_path: str = "models/action/action_model.meta.json"
    window_frames: int = 32               # sliding window fed to the temporal model
    window_stride: int = 2
    learned_min_confidence: float = 0.60
    prefer_learned: bool = True
    stability_frames: int = 4

    # --- outputs -------------------------------------------------------
    voice_enabled: bool = True
    voice_rate: int = 165
    voice_volume: float = 1.0
    voice_cooldown_s: float = 2.5

    log_dir: str = "logs"
    report_dir: str = "logs"

    record_enabled: bool = True
    record_dir: str = "outputs/recordings"
    record_fps: int = 20
    record_annotated: bool = True

    stream_enabled: bool = False
    stream_host: str = "0.0.0.0"          # bind address of the MJPEG server
    stream_port: int = 8090
    stream_quality: int = 70
    stream_target_ip: str = ""            # informational: the ground-station IP
    stream_push_url: str = ""             # optional ffmpeg push, e.g. rtsp://ip:8554/bas

    # --- files ---------------------------------------------------------
    protocol_path: str = "configs/protocol.yaml"
    zones_path: str = "configs/zones.json"

    root: Path = field(default_factory=project_root)

    # --- resolved path helpers ----------------------------------------
    @property
    def protocol_file(self) -> Path:
        return _resolve(self.protocol_path, self.root)

    @property
    def zones_file(self) -> Path:
        return _resolve(self.zones_path, self.root)

    @property
    def action_model_file(self) -> Path:
        return _resolve(self.action_model_path, self.root)

    @property
    def action_metadata_file(self) -> Path:
        return _resolve(self.action_metadata_path, self.root)

    @property
    def objects_model_file(self) -> Path:
        return _resolve(self.objects_model_path, self.root)

    @property
    def log_directory(self) -> Path:
        return _resolve(self.log_dir, self.root)

    @property
    def record_directory(self) -> Path:
        return _resolve(self.record_dir, self.root)

    def resolved_source(self) -> Any:
        """Webcam indices stay ints; everything else becomes a path/url string."""
        src = self.video_source
        if isinstance(src, int):
            return src
        text = str(src).strip()
        if text.isdigit():
            return int(text)
        if "://" in text:
            return text
        return str(_resolve(text, self.root))

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            if f.name == "root":
                continue
            out[f.name] = getattr(self, f.name)
        return out


def load_config(path: str | Path | None = None) -> AppConfig:
    root = project_root()
    path = Path(path) if path else root / "configs" / "app.yaml"
    if not path.is_absolute():
        path = root / path
    data: dict = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(AppConfig)} - {"root"}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
    cfg = AppConfig(root=root, **{k: v for k, v in data.items() if k in known})
    return cfg


def save_config(cfg: AppConfig, path: str | Path | None = None) -> Path:
    path = Path(path) if path else cfg.root / "configs" / "app.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    return path
