"""Action recognition: a heuristic tier that works immediately, a learned tier
that works better, and a fusion policy that picks between them.

The tiering is the point. A hackathon build that only works *after* you have
labelled a dataset is a build that does not work. This one runs the moment you
plug in a webcam, and gets measurably better as you feed it your own recordings:

  Tier 0  HeuristicRecognizer   zones + grip + dwell        no training
  Tier 1  LearnedRecognizer     GRU over rack features      ~30 min of clips
  Tier 2  + object detections   custom YOLO classes         optional

Tier 1 supersedes Tier 0 only when it is confident; otherwise the heuristic keeps
the session moving. The active tier is surfaced in the GUI so a jury can see
exactly which evidence path validated each step.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from aegis.perception.features import FeatureExtractor, WindowBuffer
from aegis.perception.landmarks import PerceptionResult
from aegis.perception.zones import ZoneSet
from aegis.protocol.spec import Protocol, Step

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None  # type: ignore


@dataclass
class Recognition:
    action: str | None
    confidence: float
    source: str            # "heuristic" | "learned" | "none"
    zone: str | None = None
    hand: str | None = None
    detail: dict | None = None


class HeuristicRecognizer:
    """Zone-dwell + grip-state rules derived directly from the protocol.

    For each step the protocol already declares *where* it happens (``zone``) and
    which hand. The rule is: a hand inside the expected zone, moving slowly, with
    a grip transition consistent with the action verb, for ``dwell_s`` seconds.

    Verb families recognised from the action label:
      ``grasp``/``pick``/``retrieve``/``open``  -> aperture closes (open -> shut)
      ``insert``/``place``/``secure``/``close``  -> hand still, inside zone
      ``release``/``remove``/``withdraw``        -> aperture opens (shut -> open)
      anything else                              -> presence + dwell only
    """

    CLOSE_VERBS = ("grasp", "pick", "retrieve", "grip", "hold", "take", "open")
    STILL_VERBS = ("insert", "place", "secure", "close", "seal", "press", "connect", "mount", "lock", "confirm", "transfer", "touch", "tap")
    OPEN_VERBS = ("release", "remove", "withdraw", "drop", "detach", "let_go", "stow")

    def __init__(self, protocol: Protocol, zones: ZoneSet, *, fps: float = 30.0) -> None:
        self.protocol = protocol
        self.zones = zones
        self.fps = max(1.0, fps)
        self._grip_hist: dict[str, deque] = {"left": deque(maxlen=14), "right": deque(maxlen=14)}
        self._zone_hist: dict[str, deque] = {"left": deque(maxlen=14), "right": deque(maxlen=14)}

    @staticmethod
    def _family(action: str) -> str:
        low = action.lower()
        for verb in HeuristicRecognizer.CLOSE_VERBS:
            if verb in low:
                return "close"
        for verb in HeuristicRecognizer.STILL_VERBS:
            if verb in low:
                return "still"
        for verb in HeuristicRecognizer.OPEN_VERBS:
            if verb in low:
                return "open"
        return "presence"

    def update_history(self, result: PerceptionResult) -> None:
        for side in ("left", "right"):
            hand = result.hand(side)
            if hand is None or hand.points_rack is None:
                self._grip_hist[side].append(None)
                self._zone_hist[side].append(None)
                continue
            self._grip_hist[side].append(hand.grip_aperture())
            centre = np.asarray(hand.points_rack, dtype=np.float32).mean(axis=0)
            self._zone_hist[side].append(self.zones.locate(centre))

    def _grip_trend(self, side: str) -> float:
        """Negative when the hand is closing, positive when opening."""
        vals = [v for v in self._grip_hist[side] if v is not None]
        if len(vals) < 6:
            return 0.0
        first = float(np.mean(vals[: len(vals) // 2]))
        last = float(np.mean(vals[len(vals) // 2 :]))
        return last - first

    def _zone_dwell(self, side: str, zone: str | None) -> float:
        """Fraction of recent history the hand spent in ``zone``."""
        hist = list(self._zone_hist[side])
        if not hist:
            return 0.0
        if zone is None:
            return float(sum(1 for z in hist if z is not None) / len(hist))
        return float(sum(1 for z in hist if z == zone) / len(hist))

    def score_step(self, step: Step, result: PerceptionResult) -> tuple[float, str | None, str | None]:
        """Confidence in [0, 1] that ``step`` is being performed right now."""
        family = self._family(step.action)
        sides = ("left", "right") if step.hand in ("any", "both") else (step.hand,)
        best = 0.0
        best_side: str | None = None
        best_zone: str | None = None

        for side in sides:
            hand = result.hand(side)
            if hand is None or hand.points_rack is None:
                continue
            centre = np.asarray(hand.points_rack, dtype=np.float32).mean(axis=0)
            here = self.zones.locate(centre)

            if step.zone:
                zone_obj = self.zones.get(step.zone)
                if zone_obj is None:
                    zone_score = 0.4                      # zone declared but not calibrated
                else:
                    dist = zone_obj.distance(centre)
                    zone_score = float(np.exp(-8.0 * dist))
                dwell = self._zone_dwell(side, step.zone)
            else:
                zone_score = 0.6
                dwell = self._zone_dwell(side, None)

            trend = self._grip_trend(side)
            aperture = hand.grip_aperture()
            if family == "close":
                motion = float(np.clip(0.5 - trend * 6.0, 0.0, 1.0))
                motion *= float(np.clip(1.4 - aperture, 0.15, 1.0))
            elif family == "open":
                motion = float(np.clip(0.5 + trend * 6.0, 0.0, 1.0))
                motion *= float(np.clip(aperture / 1.4, 0.15, 1.0))
            elif family == "still":
                speed = 0.0
                vals = [v for v in self._zone_hist[side] if v is not None]
                stability = len(vals) / max(1, len(self._zone_hist[side]))
                motion = float(np.clip(0.35 + 0.65 * stability - abs(trend) * 3.0, 0.0, 1.0))
            else:
                motion = 0.6

            score = 0.5 * zone_score + 0.3 * motion + 0.2 * dwell
            if score > best:
                best, best_side, best_zone = score, side, here

        return float(np.clip(best, 0.0, 1.0)), best_side, best_zone

    def recognise(self, result: PerceptionResult, expected: Step | None, protocol_window: Sequence[Step]) -> Recognition:
        """Score the expected step plus its neighbours, return the argmax.

        Scoring neighbours (not just the expected step) is what allows the
        protocol engine to notice a *skip*: if step 4 scores higher than step 2,
        the operator has jumped ahead and the engine will say so.
        """
        self.update_history(result)
        if not result.has_hands:
            return Recognition(None, 0.0, "heuristic")

        scored: list[tuple[float, Step, str | None, str | None]] = []
        for step in protocol_window:
            conf, side, zone = self.score_step(step, result)
            scored.append((conf, step, side, zone))
        if not scored:
            return Recognition(None, 0.0, "heuristic")

        scored.sort(key=lambda item: item[0], reverse=True)
        conf, step, side, zone = scored[0]

        # bias toward the expected step on near-ties: skipping is the exception,
        # so it must clear a margin before we report it. Past steps require a larger
        # margin (0.25) to avoid transient false out-of-sequence alerts.
        if expected is not None and step.id != expected.id:
            exp = next((s for s in scored if s[1].id == expected.id), None)
            if exp is not None:
                margin = 0.25 if step.id < expected.id else 0.12
                if conf - exp[0] < margin:
                    conf, step, side, zone = exp

        if conf < 0.35:
            return Recognition(None, conf, "heuristic")
        return Recognition(step.action, conf, "heuristic", zone=zone, hand=side)


class LearnedRecognizer:
    """ONNX temporal classifier over the sliding feature window."""

    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.session = None
        self.labels: list[str] = []
        self.window: int = 32
        self.dimension: int = 0
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.feature_version: int = 0
        self.error: str = ""
        self._input_name = ""
        self._load()

    @property
    def available(self) -> bool:
        return self.session is not None

    def _load(self) -> None:
        if ort is None:
            self.error = "onnxruntime not installed"
            return
        if not self.model_path.exists():
            self.error = f"no model at {self.model_path.name}"
            return
        if not self.metadata_path.exists():
            self.error = f"no metadata at {self.metadata_path.name}"
            return
        try:
            meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.labels = list(meta["labels"])
            self.window = int(meta["window"])
            self.dimension = int(meta["dimension"])
            self.feature_version = int(meta.get("feature_version", 0))
            if meta.get("mean") is not None:
                self.mean = np.asarray(meta["mean"], dtype=np.float32)
            if meta.get("std") is not None:
                self.std = np.asarray(meta["std"], dtype=np.float32)
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            self._input_name = self.session.get_inputs()[0].name
        except Exception as exc:  # pragma: no cover
            self.error = f"{type(exc).__name__}: {exc}"
            self.session = None

    def compatible_with(self, dimension: int, feature_version: int) -> bool:
        return self.available and self.dimension == dimension and self.feature_version == feature_version

    def predict(self, window: np.ndarray) -> Recognition:
        if self.session is None:
            return Recognition(None, 0.0, "learned")
        x = np.asarray(window, dtype=np.float32)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / np.where(self.std < 1e-6, 1.0, self.std)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)[None, ...]
        try:
            out = self.session.run(None, {self._input_name: x})[0][0]
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("learned inference failed: %s", exc)
            return Recognition(None, 0.0, "learned")
        probs = _softmax(np.asarray(out, dtype=np.float32))
        idx = int(np.argmax(probs))
        label = self.labels[idx] if idx < len(self.labels) else None
        if label in ("idle", "none", "background", "_idle"):
            return Recognition(None, float(probs[idx]), "learned")
        return Recognition(label, float(probs[idx]), "learned",
                           detail={"probs": {l: round(float(p), 4) for l, p in zip(self.labels, probs)}})


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


class StabilityFilter:
    """Requires N consecutive agreeing frames before an action is reported.

    Debouncing at this layer keeps the protocol engine simple and stops a single
    noisy frame from being logged as a completed step.
    """

    def __init__(self, frames: int = 4) -> None:
        self.frames = max(1, frames)
        self._buf: deque = deque(maxlen=self.frames)

    def push(self, rec: Recognition) -> Recognition:
        self._buf.append(rec)
        actions = [r.action for r in self._buf]
        if len(self._buf) < self.frames:
            return Recognition(None, rec.confidence, rec.source, rec.zone, rec.hand)
        counts = Counter(a for a in actions if a is not None)
        if not counts:
            return Recognition(None, 0.0, rec.source)
        label, hits = counts.most_common(1)[0]
        if hits < self.frames:
            return Recognition(None, rec.confidence, rec.source, rec.zone, rec.hand)
        matching = [r for r in self._buf if r.action == label]
        conf = float(np.mean([r.confidence for r in matching]))
        last = matching[-1]
        return Recognition(label, conf, last.source, last.zone, last.hand, last.detail)

    def reset(self) -> None:
        self._buf.clear()


class RecognitionEngine:
    """Owns feature extraction, both recognisers, and the fusion policy."""

    def __init__(
        self,
        protocol: Protocol,
        zones: ZoneSet,
        *,
        window_frames: int = 32,
        fps: float = 30.0,
        learned: LearnedRecognizer | None = None,
        learned_min_confidence: float = 0.60,
        prefer_learned: bool = True,
        stability_frames: int = 4,
        neighbourhood: int = 2,
    ) -> None:
        self.protocol = protocol
        self.zones = zones
        self.features = FeatureExtractor(zones, fps=fps)
        self.window = WindowBuffer(window_frames, self.features.dimension)
        self.heuristic = HeuristicRecognizer(protocol, zones, fps=fps)
        self.learned = learned
        self.learned_min_confidence = learned_min_confidence
        self.prefer_learned = prefer_learned
        self.stability = StabilityFilter(stability_frames)
        self.neighbourhood = neighbourhood
        self.last_features: np.ndarray | None = None
        self.last_raw: Recognition | None = None

    @property
    def tier(self) -> str:
        if self.learned is not None and self.learned.compatible_with(
            self.features.dimension, __import__("aegis.perception.features", fromlist=["FEATURE_VERSION"]).FEATURE_VERSION
        ):
            return "Tier 1 - learned + heuristic"
        return "Tier 0 - heuristic"

    def tier_detail(self) -> str:
        if self.learned is None:
            return "no action model loaded"
        if not self.learned.available:
            return self.learned.error or "model unavailable"
        from aegis.perception.features import FEATURE_VERSION

        if self.learned.dimension != self.features.dimension:
            return (
                f"model expects {self.learned.dimension} features, "
                f"current zone set yields {self.features.dimension} - retrain"
            )
        if self.learned.feature_version != FEATURE_VERSION:
            return f"model feature v{self.learned.feature_version} != runtime v{FEATURE_VERSION} - retrain"
        return f"{len(self.learned.labels)} classes, window {self.learned.window}"

    def protocol_window(self, cursor: int) -> list[Step]:
        lo = max(0, cursor - self.neighbourhood)
        hi = min(len(self.protocol), cursor + self.neighbourhood + 1)
        return list(self.protocol.steps[lo:hi])

    def reset(self) -> None:
        self.features.reset()
        self.window.clear()
        self.stability.reset()

    def process(self, result: PerceptionResult, cursor: int, rack_confidence: float = 0.5) -> Recognition:
        from aegis.perception.features import FEATURE_VERSION

        vec = self.features.extract(result, rack_confidence)
        self.last_features = vec
        self.window.push(vec)

        expected = self.protocol[cursor] if cursor < len(self.protocol) else None
        heur = self.heuristic.recognise(result, expected, self.protocol_window(cursor))

        chosen = heur
        if (
            self.prefer_learned
            and self.learned is not None
            and self.learned.compatible_with(self.features.dimension, FEATURE_VERSION)
            and self.window.filled
        ):
            pred = self.learned.predict(self.window.array())
            if pred.action is not None and pred.confidence >= self.learned_min_confidence:
                chosen = Recognition(
                    pred.action,
                    pred.confidence,
                    "learned",
                    zone=heur.zone,
                    hand=heur.hand,
                    detail=pred.detail,
                )
            elif pred.action is None and pred.confidence >= 0.85:
                chosen = Recognition(None, pred.confidence, "learned")

        self.last_raw = chosen
        return self.stability.push(chosen)
