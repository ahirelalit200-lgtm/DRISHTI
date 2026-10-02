"""Timestamped, structured, lightweight experiment log.

The statement asks for "a timestamped and structured lightweight text file of the
conducted steps with outcomes/status". Three artefacts are produced per session,
all plain text, all flushed line-by-line so a power loss costs at most one line:

* ``<session>.log``    fixed-width human-readable log -- what a jury reads
* ``<session>.jsonl``  one JSON object per event -- what downlink parses
* ``<session>_report.txt``  end-of-run summary table with per-step outcomes

Fixed-width columns rather than CSV because the primary consumer is a human
reading it on a console, and it stays parseable with ``split()`` anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform
import threading
import time

LOGGER = logging.getLogger(__name__)

HEADER = (
    "# VIKRAM 1 EXPERIMENT LOG\n"
    "# {experiment}\n"
    "# session : {session}\n"
    "# started : {started}\n"
    "# host    : {host}\n"
    "# steps   : {steps}\n"
    "# columns : UTC_TIMESTAMP | ELAPSED | EVENT | STEP | STATUS | CONF | DETAIL\n"
    "{rule}\n"
)

COLUMNS = "{ts:<24}| {elapsed:>9} | {event:<20}| {step:<26}| {status:<10}| {conf:>5} | {detail}"


@dataclass
class LogEntry:
    utc: str
    elapsed_s: float
    event: str
    step_id: int | None
    step_name: str
    status: str
    confidence: float | None
    detail: str


class Logbook:
    """Writes the session log. Thread-safe; one lock, short critical sections."""

    def __init__(self, directory: str | Path, session_id: str, experiment: str, total_steps: int) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.experiment = experiment
        self.total_steps = total_steps
        self.text_path = self.directory / f"{session_id}.log"
        self.jsonl_path = self.directory / f"{session_id}.jsonl"
        self.report_path = self.directory / f"{session_id}_report.txt"
        self.started_wall = time.time()
        self.started_mono = time.monotonic()
        self.entries: list[LogEntry] = []
        self._lock = threading.Lock()
        self._text = None
        self._jsonl = None
        self._open()

    def _open(self) -> None:
        self._text = self.text_path.open("w", encoding="utf-8", buffering=1)
        self._jsonl = self.jsonl_path.open("w", encoding="utf-8", buffering=1)
        started = datetime.fromtimestamp(self.started_wall, tz=timezone.utc).isoformat(timespec="seconds")
        self._text.write(
            HEADER.format(
                experiment=self.experiment,
                session=self.session_id,
                started=started,
                host=f"{platform.node()} / {platform.system()} {platform.release()}",
                steps=self.total_steps,
                rule="-" * 118,
            )
        )
        self._jsonl.write(
            json.dumps(
                {
                    "type": "session_header",
                    "session": self.session_id,
                    "experiment": self.experiment,
                    "started_utc": started,
                    "total_steps": self.total_steps,
                    "host": platform.node(),
                    "platform": f"{platform.system()} {platform.release()}",
                }
            )
            + "\n"
        )

    # ------------------------------------------------------------------ API

    @staticmethod
    def _fmt_elapsed(seconds: float) -> str:
        seconds = max(0.0, seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(int(m), 60)
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    def write_event(self, event) -> LogEntry:
        """Record a :class:`ProtocolEvent`."""
        now = time.time()
        elapsed = time.monotonic() - self.started_mono
        severity = getattr(getattr(event, "severity", None), "value", "info")
        status = {
            "step_completed": "OK",
            "step_skipped": "SKIPPED",
            "step_skipped_manual": "SKIPPED",
            "safety_block": "BLOCKED",
            "out_of_sequence": "VIOLATION",
            "unexpected_action": "VIOLATION",
            "step_timeout": "TIMEOUT",
            "wrong_zone": "VIOLATION",
            "blocked_action": "BLOCKED",
            "block_cleared": "CLEARED",
            "session_completed": "COMPLETE",
            "session_started": "START",
            "step_armed": "ARMED",
        }.get(getattr(event, "kind", ""), severity.upper())

        detail = getattr(event, "message", "")
        extra = getattr(event, "detail", None)
        if extra:
            detail = f"{detail} :: {json.dumps(extra, default=str)}"

        entry = LogEntry(
            utc=datetime.fromtimestamp(now, tz=timezone.utc).isoformat(timespec="milliseconds"),
            elapsed_s=round(elapsed, 3),
            event=getattr(event, "kind", "event"),
            step_id=getattr(event, "step_id", None),
            step_name=getattr(event, "step_name", "") or "",
            status=status,
            confidence=getattr(event, "confidence", None),
            detail=detail,
        )
        self._append(entry)
        return entry

    def write_note(self, event: str, detail: str, status: str = "INFO") -> LogEntry:
        entry = LogEntry(
            utc=datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds"),
            elapsed_s=round(time.monotonic() - self.started_mono, 3),
            event=event,
            step_id=None,
            step_name="",
            status=status,
            confidence=None,
            detail=detail,
        )
        self._append(entry)
        return entry

    def _append(self, entry: LogEntry) -> None:
        step_label = f"{entry.step_id:02d} {entry.step_name}"[:25] if entry.step_id is not None else "-"
        line = COLUMNS.format(
            ts=entry.utc,
            elapsed=self._fmt_elapsed(entry.elapsed_s),
            event=entry.event,
            step=step_label,
            status=entry.status,
            conf="-" if entry.confidence is None else f"{entry.confidence:.2f}",
            detail=entry.detail,
        )
        with self._lock:
            self.entries.append(entry)
            try:
                if self._text is not None:
                    self._text.write(line + "\n")
                if self._jsonl is not None:
                    self._jsonl.write(json.dumps({"type": "event", **asdict(entry)}, default=str) + "\n")
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("logbook write failed: %s", exc)

    # --------------------------------------------------------------- report

    def finalise(self, engine_snapshot: dict, extras: dict | None = None) -> Path:
        """Write the end-of-session summary and close the streams."""
        finished = time.time()
        duration = finished - self.started_wall
        steps = engine_snapshot.get("steps", [])
        done = sum(1 for s in steps if s["state"] == "done")
        skipped = sum(1 for s in steps if s["state"] == "skipped")
        blocked = sum(1 for s in steps if s["state"] == "blocked")
        # A bypassed safety-critical step is a failure even if the run continued.
        failed = sum(1 for s in steps if s["state"] == "failed") + blocked
        violations = sum(
            1 for e in self.entries
            if e.status in ("VIOLATION", "BLOCKED", "TIMEOUT") or e.event in ("out_of_sequence", "safety_block")
        )
        total = len(steps) or 1
        verdict = (
            "PASS" if done == total
            else "PASS WITH DEVIATIONS" if failed == 0 and done + skipped == total
            else "FAIL"
        )

        lines: list[str] = []
        lines.append("=" * 78)
        lines.append("VIKRAM 1  ::  ON-BOARD EXPERIMENT VALIDATION - EXPERIMENT SESSION REPORT")
        lines.append("=" * 78)
        lines.append(f"Experiment      : {self.experiment}")
        lines.append(f"Session ID      : {self.session_id}")
        lines.append(f"Started (UTC)   : {datetime.fromtimestamp(self.started_wall, tz=timezone.utc).isoformat(timespec='seconds')}")
        lines.append(f"Finished (UTC)  : {datetime.fromtimestamp(finished, tz=timezone.utc).isoformat(timespec='seconds')}")
        lines.append(f"Duration        : {self._fmt_elapsed(duration)}")
        lines.append(f"Verdict         : {verdict}")
        lines.append("")
        lines.append(f"Steps completed : {done}/{total}")
        lines.append(f"Steps skipped   : {skipped}")
        lines.append(f"Steps failed    : {failed}" + (f"  (incl. {blocked} bypassed safety-critical)" if blocked else ""))
        lines.append(f"Violations      : {violations}")
        for key, value in (extras or {}).items():
            lines.append(f"{key:<16}: {value}")
        lines.append("")
        lines.append("-" * 78)
        lines.append(f"{'ID':<4}{'STEP':<30}{'STATUS':<12}{'CONF':<8}{'DUR(s)':<9}NOTE")
        lines.append("-" * 78)
        for s in steps:
            lines.append(
                f"{s['id']:<4}{s['name'][:29]:<30}{s['state'].upper():<12}"
                f"{s['confidence']:<8.2f}{(s['duration_s'] if s['duration_s'] is not None else 0):<9.2f}"
                f"{s.get('note', '')}"
            )
        lines.append("-" * 78)
        lines.append("")
        lines.append("DEVIATIONS AND ALERTS")
        deviations = [
            e for e in self.entries
            if e.status in ("VIOLATION", "BLOCKED", "SKIPPED", "TIMEOUT")
        ]
        if not deviations:
            lines.append("  none - protocol executed in order")
        else:
            for e in deviations:
                lines.append(f"  [{self._fmt_elapsed(e.elapsed_s)}] {e.status:<10} {e.detail}")
        lines.append("")
        lines.append(f"Full event log  : {self.text_path.name}")
        lines.append(f"Machine log     : {self.jsonl_path.name}")
        lines.append("=" * 78)

        report = "\n".join(lines) + "\n"
        self.report_path.write_text(report, encoding="utf-8")

        with self._lock:
            try:
                if self._jsonl is not None:
                    self._jsonl.write(
                        json.dumps(
                            {
                                "type": "session_summary",
                                "verdict": verdict,
                                "duration_s": round(duration, 2),
                                "completed": done,
                                "skipped": skipped,
                                "failed": failed,
                                "violations": violations,
                                "steps": steps,
                            },
                            default=str,
                        )
                        + "\n"
                    )
                if self._text is not None:
                    self._text.write("-" * 118 + f"\n# VERDICT: {verdict} | {done}/{total} steps completed | {violations} violations\n")
            except Exception:
                pass
            for stream in (self._text, self._jsonl):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
            self._text = None
            self._jsonl = None

        LOGGER.info("session report written to %s", self.report_path)
        return self.report_path

    def tail(self, n: int = 12) -> list[LogEntry]:
        with self._lock:
            return self.entries[-n:]
