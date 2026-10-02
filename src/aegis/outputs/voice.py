from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import logging
import queue
import sys
import threading
import time

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover
    import pyttsx3
except Exception:  # pragma: no cover
    pyttsx3 = None  # type: ignore


PRIORITY_CRITICAL = 0
PRIORITY_WARNING = 1
PRIORITY_ROUTINE = 2


def _init_com() -> None:
    """Initialize COM apartment on Windows worker threads for SAPI5 stability."""
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.ole32.CoInitialize(None)
        except Exception:
            pass


@dataclass(order=True)
class Utterance:
    priority: int
    sequence: int
    text: str = field(compare=False, default="")
    state_version: int = field(compare=False, default=0)
    kind: str = field(compare=False, default="")


class VoiceAnnouncer:
    """Threaded state-aware speech queue. Safe to call from perception loop."""

    def __init__(self, *, rate: int = 200, volume: float = 1.0, cooldown_s: float = 2.5, enabled: bool = True) -> None:
        self.enabled = enabled and pyttsx3 is not None
        self.cooldown_s = cooldown_s
        self.rate = rate
        self.volume = volume
        self.available = pyttsx3 is not None
        self.error = "" if pyttsx3 is not None else "pyttsx3 not installed"
        self.spoken_count = 0
        self.last_text = ""
        self.last_event_kind = ""
        self.last_spoken_time = 0.0
        self.is_speaking = False
        self.current_state_version = 0

        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._recent: dict[str, float] = {}
        self._transcript: list[tuple[float, str]] = []

    # ------------------------------------------------------------- lifecycle

    @property
    def worker_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="voice", daemon=True)
        self._thread.start()
        LOGGER.info("[VOICE] Worker thread started")

    def _make_win32_sapi(self):
        if not sys.platform.startswith("win"):
            return None
        try:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            sapi_rate = int((self.rate - 180) / 10)
            speaker.Rate = max(-10, min(10, sapi_rate))
            speaker.Volume = int(max(0.0, min(1.0, self.volume)) * 100)
            try:
                voices = speaker.GetVoices()
                for i in range(voices.Count):
                    v = voices.Item(i)
                    desc = v.GetDescription().lower()
                    if any(tag in desc for tag in ("zira", "female", "samantha", "hazel")):
                        speaker.Voice = v
                        break
            except Exception:
                pass
            return speaker
        except Exception as exc:
            LOGGER.debug("win32 SAPI unavailable: %s", exc)
            return None

    def _make_engine(self):  # pragma: no cover - needs an audio device
        engine = pyttsx3.init() if pyttsx3 is not None else None
        if engine is None:
            return None
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", float(max(0.0, min(1.0, self.volume))))
        try:
            voices = engine.getProperty("voices")
            for v in voices:
                name = (getattr(v, "name", "") or "").lower()
                if any(tag in name for tag in ("zira", "female", "samantha", "hazel")):
                    engine.setProperty("voice", v.id)
                    break
        except Exception:
            pass
        return engine

    def _run(self) -> None:  # pragma: no cover - needs an audio device
        _init_com()
        win32_speaker = self._make_win32_sapi()
        engine = None if win32_speaker is not None else self._make_engine()

        while not self._stop.is_set():
            try:
                item: Utterance = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.enabled:
                LOGGER.info("[VOICE] DROPPED reason=VOICE_DISABLED text=\"%s\"", item.text)
                continue

            # State version check: discard background guidance if protocol state has advanced (never drop confirmed step completions)
            is_confirmed = item.kind in (
                "step_completed",
                "session_started",
                "session_completed",
                "safety_block",
                "block_cleared",
                "step_skipped",
                "step_skipped_manual",
                "speaker_test",
            )
            with self._lock:
                if not is_confirmed and item.state_version > 0 and item.state_version < self.current_state_version:
                    LOGGER.info(
                        "[VOICE] DROPPED reason=STALE_VERSION (item_ver=%d < current_ver=%d) text=\"%s\"",
                        item.state_version,
                        self.current_state_version,
                        item.text,
                    )
                    continue

            try:
                with self._lock:
                    self.is_speaking = True

                LOGGER.info(
                    "[VOICE] STARTED priority=%d kind=%s ver=%d text=\"%s\"",
                    item.priority,
                    item.kind,
                    item.state_version,
                    item.text,
                )

                if win32_speaker is not None:
                    try:
                        import pythoncom
                        pythoncom.CoInitialize()
                    except Exception:
                        pass
                    win32_speaker.Speak(item.text, 0)
                elif engine is not None:
                    engine.say(item.text)
                    engine.runAndWait()

                now = time.time()
                with self._lock:
                    self.spoken_count += 1
                    self.last_text = item.text
                    self.last_event_kind = item.kind
                    self.last_spoken_time = now
                    self._transcript.append((now, item.text))
                LOGGER.info("[VOICE] FINISHED text=\"%s\"", item.text)

            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                LOGGER.warning("[VOICE] ERROR exc=%s text=\"%s\"", exc, item.text)
                win32_speaker = self._make_win32_sapi()
                if win32_speaker is None:
                    engine = self._make_engine()
                time.sleep(0.2)
            finally:
                with self._lock:
                    self.is_speaking = False

        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------ API

    def clear_queue(self) -> None:
        """Discard all pending unspoken utterances."""
        with self._lock:
            try:
                while not self._queue.empty():
                    self._queue.get_nowait()
            except Exception:
                pass
            LOGGER.info("[VOICE] Queue cleared")

    def purge_warnings(self) -> None:
        """Purge pending out-of-sequence and warning messages from the queue."""
        with self._lock:
            keep: list[Utterance] = []
            try:
                while True:
                    item = self._queue.get_nowait()
                    if item.kind not in ("out_of_sequence", "unexpected_action", "wrong_zone"):
                        keep.append(item)
                    else:
                        LOGGER.info("[VOICE] PURGED warning text=\"%s\"", item.text)
            except queue.Empty:
                pass
            for item in keep:
                self._queue.put(item)

    def reset_voice_state(self, version: int = 0) -> None:
        """Reset voice state and clear queue for session restart or stop."""
        with self._lock:
            self.current_state_version = version
            self._recent.clear()
            self.last_text = ""
            self.last_event_kind = ""
        self.clear_queue()

    def sync_state_version(self, version: int) -> None:
        with self._lock:
            self.current_state_version = max(self.current_state_version, version)

    def status_text(self) -> str:
        if not self.enabled:
            return "OFF"
        if self.error:
            return "ERROR"
        if self.is_speaking:
            return "SPEAKING"
        return "READY"

    def say(
        self,
        text: str,
        *,
        priority: int = PRIORITY_ROUTINE,
        force: bool = False,
        state_version: int = 0,
        kind: str = "",
    ) -> bool:
        """Queue an utterance. Returns False if suppressed or stale."""
        text = (text or "").strip()
        if not text:
            return False

        if not self.enabled:
            LOGGER.info("[VOICE] DROPPED reason=VOICE_DISABLED text=\"%s\"", text)
            return False

        now = time.monotonic()
        with self._lock:
            if state_version > 0:
                self.current_state_version = max(self.current_state_version, state_version)
            elif state_version == 0:
                state_version = self.current_state_version

            is_confirmed = force or kind in (
                "step_completed",
                "session_started",
                "session_completed",
                "safety_block",
                "block_cleared",
                "step_skipped",
                "step_skipped_manual",
                "speaker_test",
            )

            if not is_confirmed and state_version > 0 and state_version < self.current_state_version:
                LOGGER.info(
                    "[VOICE] DROPPED reason=STALE_VERSION (event_ver=%d < current_ver=%d) text=\"%s\"",
                    state_version,
                    self.current_state_version,
                    text,
                )
                return False

            last = self._recent.get(text, -1e9)
            cooldown = 0.8 if (force and priority > PRIORITY_CRITICAL) else (self.cooldown_s if priority > PRIORITY_CRITICAL else 0.5)
            if (now - last) < cooldown:
                LOGGER.info("[VOICE] DROPPED reason=DUPLICATE_COOLDOWN (elapsed=%.1fs < %.1fs) text=\"%s\"", now - last, cooldown, text)
                return False

            self._recent[text] = now
            self._seq += 1
            seq = self._seq

        if kind in ("step_completed", "session_completed", "block_cleared"):
            self.purge_warnings()
            self._drain_lower_priority(PRIORITY_WARNING)
        elif priority in (PRIORITY_CRITICAL, PRIORITY_WARNING):
            self._drain_lower_priority(priority)

        self._queue.put(Utterance(priority, seq, text, state_version, kind))
        LOGGER.info("[VOICE] QUEUED priority=%d ver=%d kind=%s text=\"%s\"", priority, state_version, kind, text)
        return True

    def _drain_lower_priority(self, max_priority_to_keep: int) -> None:
        keep: list[Utterance] = []
        try:
            while True:
                item = self._queue.get_nowait()
                if item.priority <= max_priority_to_keep:
                    keep.append(item)
        except queue.Empty:
            pass
        for item in keep:
            self._queue.put(item)

    def announce_event(self, event) -> bool:
        """Map a ProtocolEvent onto a state-synchronized speech priority."""
        speech = getattr(event, "speech", None)
        if not speech or not self.enabled:
            return False

        severity = getattr(getattr(event, "severity", None), "value", "info")
        priority = {
            "critical": PRIORITY_CRITICAL,
            "warning": PRIORITY_WARNING,
        }.get(severity, PRIORITY_ROUTINE)

        kind = getattr(event, "kind", "")
        state_version = getattr(event, "state_version", 0)

        force_speak = (priority in (PRIORITY_CRITICAL, PRIORITY_WARNING)) or kind in (
            "step_completed",
            "safety_block",
            "block_cleared",
            "session_started",
            "session_completed",
            "step_skipped",
            "step_skipped_manual",
            "out_of_sequence",
            "unexpected_action",
            "wrong_zone",
            "blocked_action",
            "skip_refused",
        )

        LOGGER.info("[VOICE] EVENT_CREATED kind=%s ver=%d speech=\"%s\"", kind, state_version, speech)

        return self.say(
            speech,
            priority=priority,
            force=force_speak,
            state_version=state_version,
            kind=kind,
        )

    def test_speech_sequence(self) -> None:
        """Speak a sequence of diagnostic test phrases through the real speaker."""
        self.start()
        phrases = [
            "VIKRAM 1 voice system is working.",
            "Step 1 completed. Please pick the sample.",
            "Step 2 completed. Please transfer the sample to the tray.",
            "Session test sequence verified.",
        ]
        for phrase in phrases:
            self.say(phrase, priority=PRIORITY_WARNING, force=True, kind="speaker_test")

    def transcript(self) -> list[tuple[float, str]]:
        with self._lock:
            return list(self._transcript)

    def pending(self) -> int:
        return self._queue.qsize()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None
        LOGGER.info("[VOICE] Worker thread stopped")

