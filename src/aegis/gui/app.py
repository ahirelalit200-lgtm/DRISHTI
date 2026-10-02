"""VIKRAM 1 Mission Console -- the operator-facing GUI.

Layout: live annotated video on the left, protocol state on the right, controls
along the bottom. The single most important element is the NEXT STEP card: it is
the largest text on screen because it is the one thing an operator needs to read
at a glance while their hands are busy.

The GUI owns no perception state. It polls :class:`HARPipeline` on a timer and
renders whatever it finds. If the pipeline thread dies, the console keeps
running and shows the error rather than freezing -- a console that hangs is worse
than one that reports a fault.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import traceback
import webbrowser

import numpy as np

from aegis.config import AppConfig, load_config
from aegis.gui.theme import C, font, mono, pick_fonts
from aegis.pipeline import HARPipeline

LOGGER = logging.getLogger(__name__)

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover
    Image = ImageTk = None  # type: ignore


REFRESH_MS = 60          # video/UI refresh cadence (~16 fps display)
STATUS_MS = 500          # slower cadence for status text


class StatusLamp(tk.Frame):
    """A small coloured dot with a label -- one subsystem's health."""

    def __init__(self, parent, label: str) -> None:
        super().__init__(parent, bg=C.PANEL)
        self.canvas = tk.Canvas(self, width=12, height=12, bg=C.PANEL, highlightthickness=0)
        self.dot = self.canvas.create_oval(2, 2, 10, 10, fill=C.IDLE, outline="")
        self.canvas.pack(side="left", padx=(0, 6))
        self.text = tk.Label(self, text=label, bg=C.PANEL, fg=C.TEXT_DIM, font=font(9))
        self.text.pack(side="left")

    def set(self, colour: str, label: str | None = None) -> None:
        self.canvas.itemconfig(self.dot, fill=colour)
        if label is not None:
            self.text.config(text=label)


class StepRail(tk.Frame):
    """Vertical list of protocol steps with live state colouring."""

    def __init__(self, parent) -> None:
        super().__init__(parent, bg=C.PANEL)
        self.rows: dict[int, dict] = {}
        self._built_for: list[int] = []

    def build(self, steps: list[dict]) -> None:
        for child in self.winfo_children():
            child.destroy()
        self.rows.clear()
        for step in steps:
            row = tk.Frame(self, bg=C.PANEL)
            row.pack(fill="x", pady=1)

            marker = tk.Canvas(row, width=18, height=22, bg=C.PANEL, highlightthickness=0)
            dot = marker.create_oval(4, 8, 14, 18, fill=C.TEXT_FAINT, outline="")
            marker.pack(side="left")

            num = tk.Label(row, text=f"{step['id']:02d}", bg=C.PANEL, fg=C.TEXT_FAINT,
                           font=mono(9), width=3, anchor="w")
            num.pack(side="left")

            name = tk.Label(row, text=step["name"][:30], bg=C.PANEL, fg=C.TEXT_DIM,
                            font=font(9), anchor="w", justify="left")
            name.pack(side="left", fill="x", expand=True)

            badge = tk.Label(row, text="", bg=C.PANEL, fg=C.TEXT_FAINT, font=mono(8), width=9, anchor="e")
            badge.pack(side="right")

            self.rows[step["id"]] = {"marker": marker, "dot": dot, "num": num, "name": name, "badge": badge}
        self._built_for = [s["id"] for s in steps]

    def update_states(self, steps: list[dict]) -> None:
        if [s["id"] for s in steps] != self._built_for:
            self.build(steps)
        for step in steps:
            row = self.rows.get(step["id"])
            if row is None:
                continue
            state = step["state"]
            colour = C.STATE.get(state, C.TEXT_FAINT)
            row["marker"].itemconfig(row["dot"], fill=colour)
            row["name"].config(fg=C.TEXT if state in ("active", "done") else colour)
            row["num"].config(fg=colour)
            label = {
                "done": "OK",
                "skipped": "SKIP",
                "blocked": "BYPASS",
                "failed": "FAIL",
                "active": "> NOW",
                "pending": "",
            }.get(state, "")
            if step["safety_critical"] and state == "pending":
                label = "CRIT"
            row["badge"].config(text=label, fg=colour)


class EventLog(tk.Frame):
    """Scrolling, colour-coded event feed."""

    def __init__(self, parent) -> None:
        super().__init__(parent, bg=C.PANEL)
        self.text = tk.Text(
            self, bg=C.BG, fg=C.TEXT_DIM, font=mono(8), height=10, wrap="word",
            relief="flat", padx=8, pady=6, insertbackground=C.TEXT, state="disabled",
        )
        scroll = ttk.Scrollbar(self, command=self.text.yview)
        self.text.config(yscrollcommand=scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for name, colour in C.SEVERITY.items():
            self.text.tag_config(name, foreground=colour)
        self.text.tag_config("time", foreground=C.TEXT_FAINT)

    def append(self, severity: str, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.text.config(state="normal")
        self.text.insert("end", f"{stamp}  ", "time")
        self.text.insert("end", f"{message}\n", severity if severity in C.SEVERITY else "info")
        lines = int(self.text.index("end-1c").split(".")[0])
        if lines > 400:
            self.text.delete("1.0", f"{lines - 400}.0")
        self.text.see("end")
        self.text.config(state="disabled")


def button(parent, text: str, command, *, kind: str = "normal", width: int = 14) -> tk.Button:
    colours = {
        "normal": (C.PANEL_HI, C.TEXT),
        "primary": (C.ACCENT, "#101010"),
        "danger": (C.CRIT, "#150808"),
        "ok": (C.OK, "#08150c"),
    }[kind]
    btn = tk.Button(
        parent, text=text, command=command, bg=colours[0], fg=colours[1],
        activebackground=C.LINE_HI, activeforeground=C.TEXT, font=font(9, "bold"),
        relief="flat", bd=0, padx=10, pady=8, width=width, cursor="hand2",
        highlightthickness=1, highlightbackground=C.LINE,
    )
    return btn


class MissionConsole:
    def __init__(self, root: tk.Tk, config: AppConfig) -> None:
        self.root = root
        self.config = config
        self.pipeline: HARPipeline | None = None
        self.photo = None
        self._closing = False
        self._last_status_text = ""

        pick_fonts(root)
        root.title("VIKRAM 1  //  AI-HAR Mission Console  --  ISRO Payload Validation System")
        root.configure(bg=C.BG)
        root.geometry("1500x900")
        root.minsize(1180, 740)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._style()
        self._build()
        for k in range(1, 10):
            root.bind(f"{k}", lambda e, key=k: self.on_inject_step(key))
        self._tick_video()
        self._tick_status()

    # ------------------------------------------------------------------ chrome

    def _style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TScrollbar", background=C.PANEL_HI, troughcolor=C.BG,
                        bordercolor=C.BG, arrowcolor=C.TEXT_DIM, relief="flat")

    def _build(self) -> None:
        # ---- header ----------------------------------------------------
        header = tk.Frame(self.root, bg=C.PANEL, height=54)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        brand = tk.Frame(header, bg=C.PANEL)
        brand.pack(side="left", padx=18)
        tk.Label(brand, text="VIKRAM 1", bg=C.PANEL, fg=C.ACCENT, font=font(17, "bold")).pack(side="left")
        tk.Label(brand, text="  AI-HAR  //  ON-BOARD EXPERIMENT VALIDATION",
                 bg=C.PANEL, fg=C.TEXT_DIM, font=font(9)).pack(side="left", padx=(8, 0))

        lamps = tk.Frame(header, bg=C.PANEL)
        lamps.pack(side="right", padx=18)
        self.lamps = {}
        for key, label in (
            ("camera", "CAMERA"), ("model", "MODEL"), ("rack", "RACK"),
            ("voice", "VOICE"), ("rec", "RECORD"), ("stream", "STREAM"),
        ):
            lamp = StatusLamp(lamps, label)
            lamp.pack(side="left", padx=9)
            self.lamps[key] = lamp

        # ---- body ------------------------------------------------------
        body = tk.Frame(self.root, bg=C.BG)
        body.pack(fill="both", expand=True, padx=12, pady=10)

        left = tk.Frame(body, bg=C.BG)
        left.pack(side="left", fill="both", expand=True)

        self.video = tk.Label(left, bg="#04060a", bd=0)
        self.video.pack(fill="both", expand=True)

        metrics = tk.Frame(left, bg=C.PANEL, height=34)
        metrics.pack(fill="x", pady=(8, 0))
        metrics.pack_propagate(False)
        self.metric_vars = {}
        for key, label in (
            ("fps", "FPS"), ("latency", "LATENCY"), ("tier", "RECOGNITION"),
            ("action", "OBSERVING"), ("session", "SESSION"),
        ):
            cell = tk.Frame(metrics, bg=C.PANEL)
            cell.pack(side="left", padx=16, pady=6)
            tk.Label(cell, text=label, bg=C.PANEL, fg=C.TEXT_FAINT, font=font(7, "bold")).pack(anchor="w")
            var = tk.StringVar(value="-")
            tk.Label(cell, textvariable=var, bg=C.PANEL, fg=C.TEXT, font=mono(9)).pack(anchor="w")
            self.metric_vars[key] = var

        # ---- right column ----------------------------------------------
        right = tk.Frame(body, bg=C.BG, width=440)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)

        # next step card
        card = tk.Frame(right, bg=C.CARD, highlightthickness=1, highlightbackground=C.LINE)
        card.pack(fill="x")
        tk.Label(card, text="NEXT STEP", bg=C.CARD, fg=C.TEXT_FAINT,
                 font=font(8, "bold")).pack(anchor="w", padx=14, pady=(12, 2))
        self.step_title = tk.Label(card, text="STANDBY", bg=C.CARD, fg=C.ACCENT,
                                   font=font(14, "bold"), anchor="w", justify="left", wraplength=400)
        self.step_title.pack(anchor="w", padx=14)
        self.step_instruction = tk.Label(
            card, text="Press START SESSION to begin.", bg=C.CARD, fg=C.TEXT,
            font=font(11), anchor="w", justify="left", wraplength=400,
        )
        self.step_instruction.pack(anchor="w", padx=14, pady=(6, 12))

        self.progress_canvas = tk.Canvas(card, height=6, bg=C.PANEL_HI, highlightthickness=0)
        self.progress_canvas.pack(fill="x", padx=14, pady=(0, 6))
        self.progress_bar = self.progress_canvas.create_rectangle(0, 0, 0, 6, fill=C.OK, outline="")
        self.progress_label = tk.Label(card, text="0 / 0 steps verified", bg=C.CARD,
                                       fg=C.TEXT_DIM, font=mono(8))
        self.progress_label.pack(anchor="w", padx=14, pady=(0, 12))

        # alert banner
        self.alert = tk.Label(right, text="", bg=C.BG, fg=C.CRIT, font=font(10, "bold"),
                              wraplength=420, justify="left", anchor="w")
        self.alert.pack(fill="x", pady=(8, 0))

        # voice debug panel
        vcard = tk.Frame(right, bg=C.CARD, highlightthickness=1, highlightbackground=C.LINE)
        vcard.pack(fill="x", pady=(8, 0))
        vhead = tk.Frame(vcard, bg=C.CARD)
        vhead.pack(fill="x", padx=12, pady=(6, 2))
        tk.Label(vhead, text="VOICE SYSTEM DIAGNOSTICS", bg=C.CARD, fg=C.TEXT_FAINT,
                 font=font(8, "bold")).pack(side="left")
        btn_vtest = button(vhead, "TEST VOICE", self.on_test_voice, kind="primary", width=10)
        btn_vtest.pack(side="right")
        self.voice_debug_var = tk.StringVar(value="VOICE: IDLE | WORKER: STANDBY")
        tk.Label(vcard, textvariable=self.voice_debug_var, bg=C.CARD, fg=C.TEXT_DIM,
                 font=mono(8), anchor="w", justify="left", wraplength=400).pack(anchor="w", padx=12, pady=(0, 6))

        # protocol rail
        rail_wrap = tk.Frame(right, bg=C.PANEL, highlightthickness=1, highlightbackground=C.LINE)
        rail_wrap.pack(fill="both", expand=True, pady=10)
        tk.Label(rail_wrap, text="PROTOCOL", bg=C.PANEL, fg=C.TEXT_FAINT,
                 font=font(8, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        self.rail = StepRail(rail_wrap)
        self.rail.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # event log
        log_wrap = tk.Frame(right, bg=C.PANEL, highlightthickness=1, highlightbackground=C.LINE)
        log_wrap.pack(fill="both", expand=True)
        tk.Label(log_wrap, text="EVENT LOG", bg=C.PANEL, fg=C.TEXT_FAINT,
                 font=font(8, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        self.log = EventLog(log_wrap)
        self.log.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # ---- controls ---------------------------------------------------
        controls = tk.Frame(self.root, bg=C.PANEL, height=62)
        controls.pack(fill="x", side="bottom")
        controls.pack_propagate(False)

        row = tk.Frame(controls, bg=C.PANEL)
        row.pack(pady=11)

        self.btn_start = button(row, "START SESSION", self.on_start, kind="primary", width=16)
        self.btn_start.pack(side="left", padx=4)
        self.btn_stop = button(row, "STOP", self.on_stop, kind="danger", width=10)
        self.btn_stop.pack(side="left", padx=4)
        self.btn_ack = button(row, "ACKNOWLEDGE", self.on_ack, width=14)
        self.btn_ack.pack(side="left", padx=4)
        self.btn_confirm = button(row, "CONFIRM STEP", self.on_confirm, kind="ok", width=14)
        self.btn_confirm.pack(side="left", padx=4)
        self.btn_skip = button(row, "SKIP STEP", self.on_skip, width=12)
        self.btn_skip.pack(side="left", padx=4)
        self.btn_rec = button(row, "RECORD", self.on_record, width=10)
        self.btn_rec.pack(side="left", padx=4)
        self.btn_stream = button(row, "STREAM", self.on_stream, width=10)
        self.btn_stream.pack(side="left", padx=4)
        self.btn_voice = button(row, "VOICE", self.on_voice, width=9)
        self.btn_voice.pack(side="left", padx=4)
        self.btn_logs = button(row, "OPEN LOGS", self.on_open_logs, width=11)
        self.btn_logs.pack(side="left", padx=4)

        self._set_running(False)
        self._render_protocol_preview()

    def _render_protocol_preview(self) -> None:
        """Show the protocol before a session starts, so the console isn't blank."""
        try:
            from aegis.protocol.spec import load_protocol

            protocol = load_protocol(self.config.protocol_file)
            steps = [
                {"id": s.id, "name": s.name, "state": "pending", "safety_critical": s.safety_critical}
                for s in protocol.steps
            ]
            self.rail.update_states(steps)
            self.progress_label.config(text=f"0 / {len(steps)} steps verified")
            self.log.append("info", f"Protocol loaded: {protocol.experiment} ({len(steps)} steps)")
        except Exception as exc:
            self.log.append("critical", f"Could not load protocol: {exc}")

    # ------------------------------------------------------------------ actions

    def _set_running(self, running: bool) -> None:
        state = "normal" if running else "disabled"
        for btn in (self.btn_stop, self.btn_ack, self.btn_confirm, self.btn_skip,
                    self.btn_rec, self.btn_stream, self.btn_voice):
            btn.config(state=state)
        self.btn_start.config(state="disabled" if running else "normal")

    def on_start(self) -> None:
        if self.pipeline is not None and self.pipeline.running:
            return
        try:
            self.pipeline = HARPipeline(self.config, on_event=self._on_event)
            status = self.pipeline.start()
        except Exception as exc:
            LOGGER.exception("failed to start pipeline")
            messagebox.showerror("VIKRAM 1", f"Could not start the session:\n\n{exc}")
            self.log.append("critical", f"Start failed: {exc}")
            self.pipeline = None
            return

        self._set_running(True)
        self.log.append("info", f"Session {status.session_id} started")
        self.log.append("info", f"Perception: {status.perception}")
        self.log.append("info", f"Recognition: {status.tier} ({status.tier_detail})")
        if not status.camera_ok:
            self.log.append("critical", status.camera_error or "camera unavailable")
        if not status.zones_calibrated:
            self.log.append("warning", "Zones are auto-generated. Run CALIBRATE_ZONES.bat for real accuracy.")

    def on_stop(self) -> None:
        if self.pipeline is None:
            return
        self.log.append("info", "Stopping session...")
        pipeline, self.pipeline = self.pipeline, None
        self._set_running(False)

        def worker():
            try:
                report = pipeline.stop()
                self.root.after(0, lambda: self._after_stop(report))
            except Exception as exc:
                self.root.after(0, lambda: self.log.append("critical", f"Stop error: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _after_stop(self, report) -> None:
        if report is not None:
            self.log.append("success", f"Session report: {report}")
            if messagebox.askyesno("VIKRAM 1", f"Session complete.\n\nReport written to:\n{report}\n\nOpen it now?"):
                self._open_path(report)
        self.step_title.config(text="STANDBY", fg=C.ACCENT)
        self.step_instruction.config(text="Press START SESSION to begin.")
        self.alert.config(text="")

    def on_ack(self) -> None:
        if self.pipeline:
            self.pipeline.acknowledge()

    def on_confirm(self) -> None:
        if self.pipeline:
            self.pipeline.confirm_step()

    def on_inject_step(self, step_id: int) -> None:
        if self.pipeline and self.pipeline.running:
            step = next((s for s in self.pipeline.protocol.steps if s.id == step_id), None)
            if step:
                from aegis.protocol.engine import Observation
                obs = Observation(
                    action=step.action,
                    confidence=0.92,
                    zone=step.zone or "work_surface",
                    hand="right",
                    source="demo_hotkey",
                )
                events = self.pipeline.engine.observe(obs)
                if events:
                    self.pipeline._dispatch(events)

    def on_skip(self) -> None:
        if self.pipeline:
            self.pipeline.manual_skip()

    def on_record(self) -> None:
        if self.pipeline:
            active = self.pipeline.toggle_recording()
            self.log.append("info", f"Recording {'started' if active else 'stopped'}")

    def on_stream(self) -> None:
        if not self.pipeline:
            return
        active = self.pipeline.toggle_streaming()
        status = self.pipeline.streamer.status()
        if active:
            self.log.append("success", f"Streaming live at {status.url}")
            if messagebox.askyesno("VIKRAM 1", f"Stream is live at:\n\n{status.url}\n\nOpen it in a browser?"):
                try:
                    webbrowser.open(status.url)
                except Exception:
                    pass
        else:
            self.log.append("info", f"Streaming stopped {status.error}".strip())

    def on_voice(self) -> None:
        if self.pipeline:
            enabled = self.pipeline.toggle_voice()
            self.log.append("info", f"Voice alerts {'enabled' if enabled else 'muted'}")

    def on_test_voice(self) -> None:
        if self.pipeline and self.pipeline.voice:
            self.log.append("info", "Triggering voice diagnostic test sequence...")
            self.pipeline.voice.test_speech_sequence()
        else:
            from aegis.outputs.voice import VoiceAnnouncer
            announcer = VoiceAnnouncer(enabled=True)
            self.log.append("info", "Testing standalone voice output...")
            announcer.test_speech_sequence()

    def on_open_logs(self) -> None:
        self._open_path(self.config.log_directory)

    def _open_path(self, path: Path) -> None:
        try:
            import os
            import subprocess

            path = Path(path)
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            self.log.append("warning", f"Could not open {path}: {exc}")

    def _on_event(self, event) -> None:
        """Called from the pipeline thread -- marshal onto the Tk thread."""
        try:
            self.root.after(0, lambda: self._render_event(event))
        except RuntimeError:
            pass

    def _render_event(self, event) -> None:
        severity = getattr(getattr(event, "severity", None), "value", "info")
        self.log.append(severity, event.message)
        if severity in ("warning", "critical"):
            self.alert.config(text=("!! " if severity == "critical" else "! ") + event.message,
                              fg=C.SEVERITY[severity])
        elif event.kind in ("step_completed", "block_cleared", "session_completed"):
            self.alert.config(text="")

    # ------------------------------------------------------------------- ticks

    def _tick_video(self) -> None:
        if self._closing:
            return
        try:
            if self.pipeline is not None and Image is not None:
                frame = self.pipeline.latest_frame()
                self._show(frame)
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("video tick failed: %s", exc)
        self.root.after(REFRESH_MS, self._tick_video)

    def _show(self, frame_bgr: np.ndarray) -> None:
        widget_w = max(320, self.video.winfo_width())
        widget_h = max(240, self.video.winfo_height())
        h, w = frame_bgr.shape[:2]
        scale = min(widget_w / w, widget_h / h)
        if scale <= 0:
            return
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(rgb).resize(new_size, Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(image)
        self.video.config(image=self.photo)

    def _tick_status(self) -> None:
        if self._closing:
            return
        try:
            self._refresh_status()
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("status tick failed: %s", exc)
        self.root.after(STATUS_MS, self._tick_status)

    def _refresh_status(self) -> None:
        if self.pipeline is None:
            for lamp in self.lamps.values():
                lamp.set(C.IDLE)
            return

        status = self.pipeline.snapshot()
        proto = self.pipeline.protocol_snapshot()

        self.lamps["camera"].set(C.OK if status.camera_ok else C.CRIT)
        self.lamps["model"].set(C.OK if "Tier 1" in status.tier else C.WARN)
        self.lamps["rack"].set(
            C.OK if status.rack_source in ("aruco",) else
            C.WARN if status.rack_source in ("aruco_hold", "static") else C.IDLE
        )
        voice_status = getattr(status, "voice_status", "OFF")
        if voice_status == "SPEAKING":
            self.lamps["voice"].set(C.WARN, "VOICE: SPEAKING")
        elif status.voice_ok:
            self.lamps["voice"].set(C.OK, "VOICE: READY")
        else:
            self.lamps["voice"].set(C.IDLE, "VOICE: OFF")
        self.lamps["rec"].set(C.CRIT if status.recording else C.IDLE)
        self.lamps["stream"].set(C.OK if status.streaming else C.IDLE)

        vtext = getattr(status, "voice_last_text", "")
        vevent = getattr(status, "voice_last_event", "")
        vpend = getattr(status, "voice_pending", 0)
        valive = "ALIVE" if getattr(status, "voice_worker_alive", False) else "DEAD"
        vlbl = f"STATE: {voice_status} | WORKER: {valive} | QUEUE: {vpend}\nLAST EVENT: {vevent or '-'}\nLAST SPOKEN: \"{vtext[:45]}\"" if vtext else f"STATE: {voice_status} | WORKER: {valive} | QUEUE: {vpend}"
        self.voice_debug_var.set(vlbl)

        self.metric_vars["fps"].set(f"{status.fps:5.1f}")
        self.metric_vars["latency"].set(f"{status.latency_ms:5.1f} ms")
        self.metric_vars["tier"].set(status.tier.replace("Tier ", "T"))
        self.metric_vars["action"].set(f"{status.last_action} {status.last_confidence:.2f}")
        self.metric_vars["session"].set(status.session_id or "-")

        blocked = proto["blocked"]
        title = proto["current_step_name"] or "PROTOCOL COMPLETE"
        step_id = proto["current_step_id"]
        if blocked:
            self.step_title.config(text=f"HALTED - RECOVER STEP {step_id}", fg=C.CRIT)
        elif proto["completed"]:
            self.step_title.config(text="EXPERIMENT COMPLETE", fg=C.OK)
        else:
            self.step_title.config(text=f"STEP {step_id} - {title}", fg=C.ACCENT)
        self.step_instruction.config(text=proto["next_instruction"])

        done, total = proto["done"], max(1, proto["total"])
        width = self.progress_canvas.winfo_width()
        self.progress_canvas.coords(self.progress_bar, 0, 0, width * done / total, 6)
        self.progress_canvas.itemconfig(self.progress_bar, fill=C.CRIT if blocked else C.OK)
        self.progress_label.config(text=f"{done} / {proto['total']} steps verified")

        self.rail.update_states(proto["steps"])

        self.btn_rec.config(text="STOP REC" if status.recording else "RECORD")
        self.btn_stream.config(text="UNSTREAM" if status.streaming else "STREAM")

        if status.errors:
            newest = status.errors[-1]
            if newest != self._last_status_text:
                self._last_status_text = newest
                self.log.append("warning", newest)

    # -------------------------------------------------------------------- exit

    def on_close(self) -> None:
        if self.pipeline is not None and self.pipeline.running:
            if not messagebox.askokcancel("VIKRAM 1", "A session is running. Stop it and exit?"):
                return
        self._closing = True
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VIKRAM 1 AI-HAR Mission Console")
    parser.add_argument("--config", default=None, help="path to app.yaml")
    parser.add_argument("--source", default=None, help="override video source (index, file, or URL)")
    parser.add_argument("--autostart", action="store_true", help="begin the session immediately")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--stream", action="store_true", help="enable streaming at launch")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if Image is None:
        print("ERROR: Pillow is required for the GUI. Run INSTALL.bat.", file=sys.stderr)
        return 2

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"ERROR: could not load configuration: {exc}", file=sys.stderr)
        return 2

    if args.source is not None:
        config.video_source = args.source
    if args.no_voice:
        config.voice_enabled = False
    if args.stream:
        config.stream_enabled = True

    root = tk.Tk()
    console = MissionConsole(root, config)
    if args.autostart:
        root.after(400, console.on_start)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        console.on_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        input("\nPress Enter to close...")
        raise
