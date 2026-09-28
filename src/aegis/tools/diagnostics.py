"""Environment diagnostics -- run this first when something does not work.

Checks every subsystem the app depends on and prints a verdict per item, plus a
concrete fix for anything that fails. Designed so that the output can be pasted
into a chat or an issue and be immediately actionable.

Exit code 0 = ready to run live, 1 = runs degraded, 2 = will not run.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import platform
import socket
import sys
import time

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
SYMBOL = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, section: str, name: str, status: str, detail: str = "") -> None:
        self.rows.append((section, name, status, detail))

    def worst(self) -> str:
        if any(r[2] == FAIL for r in self.rows):
            return FAIL
        if any(r[2] == WARN for r in self.rows):
            return WARN
        return OK

    def render(self) -> str:
        lines = []
        current = None
        for section, name, status, detail in self.rows:
            if section != current:
                current = section
                lines.append("")
                lines.append(f"--- {section} " + "-" * max(0, 58 - len(section)))
            line = f"  {SYMBOL[status]}  {name}"
            if detail:
                line += f"\n            {detail}"
            lines.append(line)
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "verdict": self.worst(),
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version,
            "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "checks": [
                {"section": s, "name": n, "status": st, "detail": d} for s, n, st, d in self.rows
            ],
        }


def check_python(report: Report) -> None:
    version = sys.version_info
    if version < (3, 10):
        report.add("Python", f"Python {version.major}.{version.minor}", FAIL,
                   "Python 3.10-3.12 required. Install 3.11 from python.org.")
    elif version >= (3, 13):
        report.add("Python", f"Python {version.major}.{version.minor}", WARN,
                   "MediaPipe wheels may not exist for 3.13+. Python 3.11 is the safe choice.")
    else:
        report.add("Python", f"Python {version.major}.{version.minor}.{version.micro}", OK)
    report.add("Python", f"{platform.system()} {platform.release()} {platform.machine()}", OK)
    report.add("Python", f"64-bit interpreter", OK if sys.maxsize > 2**32 else FAIL,
               "" if sys.maxsize > 2**32 else "32-bit Python cannot run MediaPipe. Reinstall 64-bit.")


def check_packages(report: Report) -> None:
    required = [
        ("numpy", "core maths", True),
        ("cv2", "video capture, encoding, ArUco (opencv-contrib-python)", True),
        ("yaml", "config parsing (PyYAML)", True),
        ("PIL", "GUI image rendering (Pillow)", True),
    ]
    optional = [
        ("mediapipe", "hand + pose landmarks -- without it, recognition is motion-only"),
        ("onnxruntime", "runs the trained Tier-1 action model"),
        ("pyttsx3", "offline voice alerts"),
        ("onnx", "needed only to TRAIN and export a model"),
        ("torch", "optional faster training backend"),
    ]
    for module, purpose, _ in required:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "?")
            report.add("Packages", f"{module} {version}", OK, purpose)
        except Exception as exc:
            report.add("Packages", module, FAIL, f"{purpose}. Missing: {exc}. Run INSTALL.bat.")

    for module, purpose in optional:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "?")
            report.add("Packages", f"{module} {version}", OK, purpose)
        except Exception:
            report.add("Packages", module, WARN, f"not installed -- {purpose}")


def check_tk(report: Report) -> None:
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        root.destroy()
        report.add("GUI", f"tkinter {tkinter.TkVersion}", OK)
    except Exception as exc:
        report.add("GUI", "tkinter", FAIL,
                   f"{exc}. On Windows, re-run the Python installer and enable 'tcl/tk and IDLE'.")


def check_camera(report: Report, max_index: int = 4) -> None:
    try:
        from aegis.capture.source import probe_cameras
    except Exception as exc:
        report.add("Camera", "probe", FAIL, str(exc))
        return
    found = probe_cameras(max_index)
    if found:
        report.add("Camera", f"working camera indices: {found}", OK,
                   f"set video_source: {found[0]} in configs/app.yaml")
    else:
        report.add("Camera", "no camera opened", WARN,
                   "Close Teams/Zoom/OBS, check Windows camera privacy settings, "
                   "or point video_source at a video file to test.")


def check_perception(report: Report, root: Path) -> None:
    try:
        from aegis.perception.landmarks import detect_backend

        backend = detect_backend(root / "models" / "mediapipe")
    except Exception as exc:
        report.add("Perception", "landmark backend", FAIL, str(exc))
        return

    if backend == "solutions":
        report.add("Perception", "MediaPipe classic solutions", OK, "hands + pose available, no model files needed")
    elif backend == "tasks":
        report.add("Perception", "MediaPipe Tasks", OK, "hand/pose .task bundles present")
    elif backend == "tasks-missing-models":
        report.add("Perception", "MediaPipe Tasks (models missing)", WARN,
                   "This wheel has no mp.solutions. Run FETCH_MODELS.bat to download the .task files.")
    else:
        report.add("Perception", "no landmark backend", WARN,
                   "Falling back to motion-energy only. Recognition will be unreliable. "
                   "Try: pip install mediapipe==0.10.14")

    try:
        import cv2

        report.add("Perception", "cv2.aruco (rack frame)", OK if hasattr(cv2, "aruco") else WARN,
                   "" if hasattr(cv2, "aruco") else
                   "Install opencv-contrib-python for ArUco rack tracking; static calibration still works.")
    except Exception:
        pass


def check_configs(report: Report, root: Path) -> None:
    from aegis.config import load_config
    from aegis.protocol.spec import load_protocol
    from aegis.perception.zones import ZoneSet

    try:
        config = load_config()
        report.add("Configuration", "configs/app.yaml", OK)
    except Exception as exc:
        report.add("Configuration", "configs/app.yaml", FAIL, str(exc))
        return

    try:
        protocol = load_protocol(config.protocol_file)
        report.add("Configuration", f"protocol: {protocol.experiment}", OK,
                   f"{len(protocol)} steps, {len(protocol.action_labels)} distinct actions")
        critical = [s.id for s in protocol.steps if s.safety_critical]
        report.add("Configuration", f"safety-critical steps: {critical or 'none'}",
                   OK if critical else WARN,
                   "" if critical else "No step is marked safety_critical -- nothing will ever trigger a hard block.")
    except Exception as exc:
        report.add("Configuration", "protocol", FAIL, f"{exc}")
        return

    zones = ZoneSet.load(config.zones_file)
    declared = set(protocol.zone_names)
    if not declared:
        report.add("Configuration", "zones", WARN, "protocol declares no zones; recognition will be motion-only")
    elif not len(zones):
        report.add("Configuration", "zones not calibrated", WARN,
                   f"Run CALIBRATE_ZONES.bat and define: {', '.join(sorted(declared))}")
    else:
        missing = declared - set(zones.names)
        if missing:
            report.add("Configuration", f"zones calibrated ({len(zones)})", WARN,
                       f"missing polygons for: {', '.join(sorted(missing))}")
        else:
            report.add("Configuration", f"zones calibrated ({len(zones)})", OK)

    quad = config.zones_file.with_name("rack_quad.json")
    report.add("Configuration", "rack quad", OK if quad.exists() else WARN,
               "" if quad.exists() else "Not calibrated. ArUco markers will still work; otherwise the whole frame is used.")


def check_model(report: Report) -> None:
    from aegis.config import load_config
    from aegis.perception.recognizer import LearnedRecognizer

    config = load_config()
    if not config.action_model_file.exists():
        report.add("Action model", "not trained yet", WARN,
                   "System runs on the Tier-0 heuristic. Run RECORD_DATASET.bat then TRAIN_MODEL.bat.")
        return
    recogniser = LearnedRecognizer(config.action_model_file, config.action_metadata_file)
    if not recogniser.available:
        report.add("Action model", "present but unloadable", FAIL, recogniser.error)
        return
    report.add("Action model", f"{len(recogniser.labels)} classes, window {recogniser.window}", OK,
               f"features v{recogniser.feature_version}, dim {recogniser.dimension}")
    try:
        meta = json.loads(config.action_metadata_file.read_text(encoding="utf-8"))
        acc = meta.get("validation_accuracy")
        holdout = meta.get("held_out_operator")
        if acc is not None:
            status = OK if acc >= 0.75 else WARN
            report.add("Action model", f"held-out accuracy {acc:.1%} (operator {holdout})", status,
                       "" if status == OK else "Record more clips / more operators and retrain.")
    except Exception:
        pass


def check_outputs(report: Report, root: Path) -> None:
    for name, path in (("logs", root / "logs"), ("recordings", root / "outputs" / "recordings")):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            report.add("Outputs", f"{name} writable", OK, str(path))
        except Exception as exc:
            report.add("Outputs", f"{name} not writable", FAIL, f"{path}: {exc}")

    try:
        import cv2
        import numpy as np
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "probe.mp4"
            writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (160, 120))
            usable = writer.isOpened()
            if usable:
                writer.write(np.zeros((120, 160, 3), np.uint8))
            writer.release()
        report.add("Outputs", "mp4v video encoder", OK if usable else WARN,
                   "" if usable else "mp4 writing failed; the recorder will fall back to MJPEG/.avi")
    except Exception as exc:
        report.add("Outputs", "video encoder", WARN, str(exc))


def check_network(report: Report) -> None:
    from aegis.config import load_config
    from aegis.capture.streamer import VideoStreamer

    config = load_config()
    ip = VideoStreamer.local_ip()
    report.add("Streaming", f"this machine's LAN address: {ip}", OK,
               f"ground station opens http://{ip}:{config.stream_port}/")
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((config.stream_host, config.stream_port))
        probe.close()
        report.add("Streaming", f"port {config.stream_port} free", OK)
    except OSError as exc:
        report.add("Streaming", f"port {config.stream_port} unavailable", WARN,
                   f"{exc}. Change stream_port in configs/app.yaml.")

    import shutil

    if config.stream_push_url:
        has_ffmpeg = shutil.which("ffmpeg") is not None
        report.add("Streaming", "ffmpeg for push mode", OK if has_ffmpeg else WARN,
                   "" if has_ffmpeg else "stream_push_url is set but ffmpeg is not on PATH")


def check_voice(report: Report) -> None:
    try:
        import pyttsx3
    except Exception:
        report.add("Voice", "pyttsx3 not installed", WARN,
                   "Voice alerts disabled. pip install pyttsx3")
        return
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        engine.stop()
        report.add("Voice", f"speech engine ready ({len(voices)} voices)", OK)
    except Exception as exc:
        report.add("Voice", "speech engine failed to initialise", WARN,
                   f"{exc}. On Windows check Settings > Time & Language > Speech.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AEGIS environment diagnostics")
    parser.add_argument("--json", default=None, help="also write a JSON report here")
    parser.add_argument("--no-camera", action="store_true", help="skip the camera probe (it can be slow)")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "src"))

    print("=" * 66)
    print("  AEGIS AI-HAR  ::  ENVIRONMENT DIAGNOSTICS")
    print("=" * 66)

    report = Report()
    check_python(report)
    check_packages(report)
    check_tk(report)
    check_perception(report, root)
    if not args.no_camera:
        check_camera(report)
    try:
        check_configs(report, root)
        check_model(report)
        check_outputs(report, root)
        check_network(report)
    except Exception as exc:
        report.add("Configuration", "unexpected failure", FAIL, str(exc))
    check_voice(report)

    print(report.render())
    verdict = report.worst()
    print()
    print("=" * 66)
    if verdict == OK:
        print("  VERDICT: READY  --  double-click START.bat")
        code = 0
    elif verdict == WARN:
        print("  VERDICT: RUNS DEGRADED  --  the app will start, but see the WARN items above")
        code = 1
    else:
        print("  VERDICT: WILL NOT RUN  --  fix the FAIL items above, then re-run this")
        code = 2
    print("=" * 66)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
        print(f"\nJSON report: {args.json}")
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
