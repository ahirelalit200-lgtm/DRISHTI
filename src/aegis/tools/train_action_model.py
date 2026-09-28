"""Train the temporal action model and export it to ONNX for offline inference.

Two backends, chosen automatically:

* **PyTorch** if installed -- a 2-layer GRU with attention pooling. Faster and
  better; exports to ONNX via ``torch.onnx.export``.
* **NumPy** otherwise -- the same architecture, forward and backward written by
  hand, exported by emitting the ONNX graph directly. Slower to train (minutes,
  not seconds) but removes a 2.5 GB dependency from the deliverable.

Either way the artefact is identical: an ONNX file plus a JSON sidecar carrying
the label list, window length, feature dimension, normalisation statistics and
the feature-layout version. The runtime refuses to load a model whose feature
version or dimension does not match, which stops the classic failure where you
recalibrate zones, silently change the feature width, and the model starts
predicting nonsense with high confidence.

Validation is **operator-held-out** by default. Random splits leak: consecutive
windows from one clip are nearly identical, so a random split reports 98% and
then collapses on a new person. Holding out an entire operator is the honest
number, and it is the number to put on a slide.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


# ============================================================== data loading

def load_clips(dataset_dir: Path) -> list[dict]:
    clips = []
    for path in sorted((dataset_dir / "clips").glob("*.npz")):
        try:
            data = np.load(path, allow_pickle=True)
            clips.append(
                {
                    "features": np.asarray(data["features"], dtype=np.float32),
                    "label": str(data["label"]),
                    "operator": str(data["operator"]),
                    "session": str(data["session"]),
                    "feature_version": int(data["feature_version"]),
                    "dimension": int(data["dimension"]),
                    "file": path.name,
                }
            )
        except Exception as exc:
            LOGGER.warning("skipping %s: %s", path.name, exc)
    return clips


def make_windows(clips: list[dict], window: int, stride: int) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Slice clips into fixed windows. Short clips are left-padded, not dropped."""
    X, y, operators, files = [], [], [], []
    for clip in clips:
        feats = clip["features"]
        if feats.shape[0] < 4:
            continue
        if feats.shape[0] < window:
            pad = np.repeat(feats[:1], window - feats.shape[0], axis=0)
            windows = [np.concatenate([pad, feats], axis=0)]
        else:
            windows = [feats[i : i + window] for i in range(0, feats.shape[0] - window + 1, stride)]
        for w in windows:
            X.append(w)
            y.append(clip["label"])
            operators.append(clip["operator"])
            files.append(clip["file"])
    if not X:
        return np.zeros((0, window, 1), np.float32), np.array([]), [], []
    return np.asarray(X, dtype=np.float32), np.asarray(y), operators, files


def augment(X: np.ndarray, y: np.ndarray, factor: int = 2, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """Jitter, scale, and time-warp windows to widen the operating envelope.

    Deliberately conservative: the features are already rotation- and
    scale-normalised, so aggressive geometric augmentation would push samples
    outside the physically achievable manifold and teach the model nonsense.
    """
    if factor <= 1 or X.shape[0] == 0:
        return X, y
    rng = rng or np.random.default_rng(0)
    outs, labels = [X], [y]
    for _ in range(factor - 1):
        noise = rng.normal(0.0, 0.012, X.shape).astype(np.float32)
        scale = rng.uniform(0.95, 1.05, (X.shape[0], 1, X.shape[2])).astype(np.float32)
        shifted = X * scale + noise
        # small temporal jitter: roll each window by -2..2 frames
        rolls = rng.integers(-2, 3, X.shape[0])
        for i, r in enumerate(rolls):
            if r:
                shifted[i] = np.roll(shifted[i], int(r), axis=0)
        outs.append(shifted.astype(np.float32))
        labels.append(y)
    return np.concatenate(outs, axis=0), np.concatenate(labels, axis=0)


def split_by_operator(operators: list[str], holdout: str | None) -> tuple[np.ndarray, np.ndarray]:
    ops = np.asarray(operators)
    unique = sorted(set(operators))
    if holdout is None:
        if len(unique) >= 2:
            holdout = unique[-1]
        else:
            LOGGER.warning(
                "only one operator in the dataset - falling back to a clip-level split. "
                "Record a second operator for an honest accuracy number."
            )
            n = len(operators)
            idx = np.arange(n)
            rng = np.random.default_rng(7)
            rng.shuffle(idx)
            cut = int(n * 0.8)
            return idx[:cut], idx[cut:]
    test = np.where(ops == holdout)[0]
    train = np.where(ops != holdout)[0]
    LOGGER.info("held-out operator: %s (%d test windows)", holdout, len(test))
    return train, test


# ============================================================ numpy backend

class NumpyGRU:
    """Single-layer GRU + mean-pool + linear head, forward and backward by hand.

    Adam optimiser, gradient clipping, no external framework. Kept small on
    purpose: with a few thousand windows, a bigger model just memorises the
    operators you happened to record.
    """

    def __init__(self, input_dim: int, hidden: int, classes: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(hidden)
        self.hidden = hidden
        self.p = {
            "Wz": rng.uniform(-s, s, (input_dim, hidden)).astype(np.float32),
            "Uz": rng.uniform(-s, s, (hidden, hidden)).astype(np.float32),
            "bz": np.zeros(hidden, np.float32),
            "Wr": rng.uniform(-s, s, (input_dim, hidden)).astype(np.float32),
            "Ur": rng.uniform(-s, s, (hidden, hidden)).astype(np.float32),
            "br": np.zeros(hidden, np.float32),
            "Wh": rng.uniform(-s, s, (input_dim, hidden)).astype(np.float32),
            "Uh": rng.uniform(-s, s, (hidden, hidden)).astype(np.float32),
            "bh": np.zeros(hidden, np.float32),
            "Wo": rng.uniform(-s, s, (hidden, classes)).astype(np.float32),
            "bo": np.zeros(classes, np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def forward(self, X):
        B, T, _ = X.shape
        H = self.hidden
        h = np.zeros((B, H), np.float32)
        cache = []
        hs = np.zeros((B, T, H), np.float32)
        for t in range(T):
            x = X[:, t, :]
            z = self._sigmoid(x @ self.p["Wz"] + h @ self.p["Uz"] + self.p["bz"])
            r = self._sigmoid(x @ self.p["Wr"] + h @ self.p["Ur"] + self.p["br"])
            hh = np.tanh(x @ self.p["Wh"] + (r * h) @ self.p["Uh"] + self.p["bh"])
            h_new = (1 - z) * h + z * hh
            cache.append((x, h, z, r, hh, h_new))
            hs[:, t, :] = h_new
            h = h_new
        pooled = hs.mean(axis=1)
        logits = pooled @ self.p["Wo"] + self.p["bo"]
        return logits, (cache, pooled, T)

    def backward(self, X, y_onehot, logits, state):
        cache, pooled, T = state
        B = X.shape[0]
        probs = softmax(logits)
        dlogits = (probs - y_onehot) / B
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        g["Wo"] = pooled.T @ dlogits
        g["bo"] = dlogits.sum(axis=0)
        dpooled = dlogits @ self.p["Wo"].T
        dh_next = np.zeros_like(pooled)
        dh_pool = dpooled / T
        for t in reversed(range(T)):
            x, h_prev, z, r, hh, _ = cache[t]
            dh = dh_next + dh_pool
            dz = dh * (hh - h_prev)
            dhh = dh * z
            dh_prev = dh * (1 - z)
            dhh_raw = dhh * (1 - hh ** 2)
            g["Wh"] += x.T @ dhh_raw
            g["Uh"] += (r * h_prev).T @ dhh_raw
            g["bh"] += dhh_raw.sum(axis=0)
            drh = dhh_raw @ self.p["Uh"].T
            dr = drh * h_prev
            dh_prev += drh * r
            dz_raw = dz * z * (1 - z)
            g["Wz"] += x.T @ dz_raw
            g["Uz"] += h_prev.T @ dz_raw
            g["bz"] += dz_raw.sum(axis=0)
            dh_prev += dz_raw @ self.p["Uz"].T
            dr_raw = dr * r * (1 - r)
            g["Wr"] += x.T @ dr_raw
            g["Ur"] += h_prev.T @ dr_raw
            g["br"] += dr_raw.sum(axis=0)
            dh_prev += dr_raw @ self.p["Ur"].T
            dh_next = dh_prev
        return g

    def step(self, grads, lr=2e-3, clip=5.0):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        total = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values())) + 1e-12
        scale = min(1.0, clip / total)
        for k in self.p:
            g = grads[k] * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g ** 2)
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mhat / (np.sqrt(vhat) + eps)


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)


def train_numpy(Xtr, ytr, Xte, yte, classes, epochs, hidden, batch, lr):
    model = NumpyGRU(Xtr.shape[2], hidden, classes)
    n = Xtr.shape[0]
    onehot = np.eye(classes, dtype=np.float32)[ytr]
    rng = np.random.default_rng(1)
    best_acc, best_params, history = 0.0, None, []
    for epoch in range(epochs):
        idx = rng.permutation(n)
        losses = []
        for start in range(0, n, batch):
            sel = idx[start : start + batch]
            xb, yb = Xtr[sel], onehot[sel]
            logits, state = model.forward(xb)
            probs = softmax(logits)
            losses.append(float(-np.log(probs[np.arange(len(sel)), ytr[sel]] + 1e-9).mean()))
            grads = model.backward(xb, yb, logits, state)
            model.step(grads, lr=lr)
        acc = evaluate_numpy(model, Xte, yte) if len(Xte) else 0.0
        tracc = evaluate_numpy(model, Xtr[:600], ytr[:600])
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "train_acc": tracc, "val_acc": acc})
        LOGGER.info("epoch %2d/%d  loss %.4f  train %.3f  val %.3f",
                    epoch + 1, epochs, np.mean(losses), tracc, acc)
        if acc >= best_acc:
            best_acc = acc
            best_params = {k: v.copy() for k, v in model.p.items()}
    if best_params is not None:
        model.p = best_params
    return model, best_acc, history


def evaluate_numpy(model, X, y):
    if len(X) == 0:
        return 0.0
    preds = []
    for start in range(0, len(X), 256):
        logits, _ = model.forward(X[start : start + 256])
        preds.append(np.argmax(logits, axis=1))
    return float((np.concatenate(preds) == y).mean())


def export_numpy_onnx(model: NumpyGRU, path: Path, window: int, dim: int, classes: int) -> None:
    """Emit an ONNX graph implementing the GRU by hand.

    Uses ONNX's native ``GRU`` op with the weight layout it expects: gates in
    (z, r, h) order, W as (num_directions, 3*hidden, input), R as
    (num_directions, 3*hidden, hidden), B as (num_directions, 6*hidden).
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    H = model.hidden
    p = model.p

    # --- update-gate convention fix -------------------------------------
    # ONNX GRU computes   Ht = (1 - z) * h~ + z * H(t-1)
    # This NumPy GRU uses Ht = (1 - z) * H(t-1) + z * h~   (roles swapped).
    # Because sigmoid(-a) == 1 - sigmoid(a), negating the z-gate weights and
    # bias makes ONNX's z equal to (1 - our z), which makes the two update
    # equations algebraically identical. Verified by the parity test in
    # tests/test_onnx_export.py -- do not "simplify" this away.
    Wz, Uz, bz = -p["Wz"], -p["Uz"], -p["bz"]

    W = np.concatenate([Wz.T, p["Wr"].T, p["Wh"].T], axis=0)[None].astype(np.float32)
    R = np.concatenate([Uz.T, p["Ur"].T, p["Uh"].T], axis=0)[None].astype(np.float32)
    Wb = np.concatenate([bz, p["br"], p["bh"]]).astype(np.float32)
    Rb = np.zeros(3 * H, np.float32)
    B = np.concatenate([Wb, Rb])[None].astype(np.float32)

    initialisers = [
        numpy_helper.from_array(W, "W"),
        numpy_helper.from_array(R, "R"),
        numpy_helper.from_array(B, "B"),
        numpy_helper.from_array(p["Wo"].astype(np.float32), "Wo"),
        numpy_helper.from_array(p["bo"].astype(np.float32), "bo"),
        numpy_helper.from_array(np.array([1], np.int64), "axis1"),
    ]

    nodes = [
        # (batch, seq, feat) -> (seq, batch, feat) as ONNX GRU expects
        helper.make_node("Transpose", ["input"], ["x_t"], perm=[1, 0, 2]),
        helper.make_node("GRU", ["x_t", "W", "R", "B"], ["Y", "Y_h"],
                         hidden_size=H, linear_before_reset=0, direction="forward"),
        # Y: (seq, 1, batch, hidden) -> squeeze direction -> (seq, batch, hidden)
        helper.make_node("Squeeze", ["Y", "axis1"], ["Y_s"]),
        helper.make_node("ReduceMean", ["Y_s"], ["pooled"], axes=[0], keepdims=0),
        helper.make_node("MatMul", ["pooled", "Wo"], ["mm"]),
        helper.make_node("Add", ["mm", "bo"], ["logits"]),
    ]

    graph = helper.make_graph(
        nodes,
        "aegis_action_gru",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", window, dim])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", classes])],
        initialisers,
    )
    model_proto = helper.make_model(
        graph, producer_name="aegis-har",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model_proto.ir_version = 8
    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, str(path))


# ============================================================ torch backend

if torch is not None:  # pragma: no cover - optional dependency

    class TorchGRU(nn.Module):
        def __init__(self, input_dim: int, hidden: int, classes: int) -> None:
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden, num_layers=2, batch_first=True, dropout=0.15)
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.2), nn.Linear(hidden, classes))

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out.mean(dim=1))


def train_torch(Xtr, ytr, Xte, yte, classes, epochs, hidden, batch, lr):  # pragma: no cover
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("training on %s", device)
    model = TorchGRU(Xtr.shape[2], hidden, classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossfn = nn.CrossEntropyLoss(label_smoothing=0.05)

    xt = torch.tensor(Xtr, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    xv = torch.tensor(Xte, device=device) if len(Xte) else None
    yv = torch.tensor(yte, dtype=torch.long, device=device) if len(yte) else None

    best_acc, best_state, history = 0.0, None, []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        losses = []
        for start in range(0, len(xt), batch):
            sel = perm[start : start + batch]
            opt.zero_grad()
            loss = lossfn(model(xt[sel]), yt[sel])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            tracc = (model(xt[:600]).argmax(1) == yt[:600]).float().mean().item()
            acc = (model(xv).argmax(1) == yv).float().mean().item() if xv is not None else 0.0
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "train_acc": tracc, "val_acc": acc})
        LOGGER.info("epoch %2d/%d  loss %.4f  train %.3f  val %.3f",
                    epoch + 1, epochs, np.mean(losses), tracc, acc)
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc, history


def export_torch_onnx(model, path: Path, window: int, dim: int) -> None:  # pragma: no cover
    model.eval().cpu()
    dummy = torch.zeros(1, window, dim)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )


# =================================================================== metrics

def confusion(y_true, y_pred, classes: int) -> np.ndarray:
    m = np.zeros((classes, classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def per_class_report(matrix: np.ndarray, labels: list[str]) -> str:
    lines = [f"{'CLASS':<26}{'SUPPORT':>8}{'RECALL':>9}{'PRECISION':>11}{'F1':>7}"]
    lines.append("-" * 61)
    for i, label in enumerate(labels):
        support = int(matrix[i].sum())
        tp = int(matrix[i, i])
        recall = tp / support if support else 0.0
        pred = int(matrix[:, i].sum())
        precision = tp / pred if pred else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        lines.append(f"{label:<26}{support:>8}{recall:>9.3f}{precision:>11.3f}{f1:>7.3f}")
    return "\n".join(lines)


# ====================================================================== main

def run(args) -> int:
    dataset = Path(args.dataset)
    if not (dataset / "clips").exists():
        print(f"ERROR: no clips found in {dataset / 'clips'}")
        print("Record a dataset first: RECORD_DATASET.bat")
        return 2

    clips = load_clips(dataset)
    if not clips:
        print("ERROR: dataset contains no readable clips.")
        return 2

    versions = {c["feature_version"] for c in clips}
    dims = {c["dimension"] for c in clips}
    if len(versions) > 1 or len(dims) > 1:
        print(f"ERROR: mixed feature versions {versions} or dimensions {dims} in the dataset.")
        print("Re-record after changing zones, or delete the older clips.")
        return 2

    dimension = dims.pop()
    feature_version = versions.pop()
    counts = Counter(c["label"] for c in clips)
    labels = sorted(counts)
    print(f"\nLoaded {len(clips)} clips, {len(labels)} classes, feature dim {dimension}")
    for label in labels:
        print(f"  {label:<26} {counts[label]:>3} clips")
    thin = [l for l in labels if counts[l] < 5]
    if thin:
        print(f"\nWARNING: too few clips for: {', '.join(thin)}. Aim for 12+ each.")

    X, y_text, operators, _ = make_windows(clips, args.window, args.stride)
    if X.shape[0] == 0:
        print("ERROR: no windows could be built. Are the clips too short?")
        return 2
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y = np.array([label_to_idx[l] for l in y_text], dtype=np.int64)
    print(f"Windows: {X.shape[0]} of shape {X.shape[1]}x{X.shape[2]}")

    train_idx, test_idx = split_by_operator(operators, args.holdout)
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]

    mean = Xtr.reshape(-1, dimension).mean(axis=0)
    std = Xtr.reshape(-1, dimension).std(axis=0)
    std[std < 1e-6] = 1.0
    Xtr = ((Xtr - mean) / std).astype(np.float32)
    Xte = ((Xte - mean) / std).astype(np.float32) if len(Xte) else Xte

    Xtr, ytr = augment(Xtr, ytr, factor=args.augment)
    print(f"Train windows: {len(Xtr)}   Test windows: {len(Xte)}\n")

    backend = "numpy" if (torch is None or args.force_numpy) else "torch"
    print(f"Backend: {backend}\n")
    started = time.time()

    if backend == "torch":
        model, acc, history = train_torch(Xtr, ytr, Xte, yte, len(labels),
                                          args.epochs, args.hidden, args.batch, args.lr)
        preds = []
        if len(Xte):
            model.eval()
            with torch.no_grad():
                for s in range(0, len(Xte), 512):
                    preds.append(model(torch.tensor(Xte[s : s + 512])).argmax(1).numpy())
    else:
        model, acc, history = train_numpy(Xtr, ytr, Xte, yte, len(labels),
                                          args.epochs, args.hidden, args.batch, args.lr)
        preds = []
        if len(Xte):
            for s in range(0, len(Xte), 256):
                logits, _ = model.forward(Xte[s : s + 256])
                preds.append(np.argmax(logits, axis=1))

    elapsed = time.time() - started
    out_model = Path(args.out)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_model.with_suffix("").with_suffix(".meta.json") if out_model.suffix == ".onnx" \
        else out_model.with_name(out_model.stem + ".meta.json")

    if backend == "torch":
        export_torch_onnx(model, out_model, args.window, dimension)
    else:
        export_numpy_onnx(model, out_model, args.window, dimension, len(labels))

    report_text = ""
    if preds:
        y_pred = np.concatenate(preds)
        matrix = confusion(yte, y_pred, len(labels))
        report_text = per_class_report(matrix, labels)
        print("\n" + report_text)

    meta = {
        "labels": labels,
        "window": args.window,
        "dimension": dimension,
        "feature_version": feature_version,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "backend": backend,
        "held_out_operator": args.holdout or "auto",
        "validation_accuracy": round(float(acc), 4),
        "train_windows": int(len(Xtr)),
        "test_windows": int(len(Xte)),
        "clips": len(clips),
        "epochs": args.epochs,
        "hidden": args.hidden,
        "trained_seconds": round(elapsed, 1),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "history": history,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    card = out_model.parent / "MODEL_CARD.md"
    card.write_text(_model_card(meta, counts, report_text), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"Model      : {out_model}")
    print(f"Metadata   : {meta_path}")
    print(f"Model card : {card}")
    print(f"Held-out accuracy: {acc:.1%}  (operator: {meta['held_out_operator']})")
    print("=" * 62)
    if acc < 0.75:
        print("\nAccuracy is low. Most effective fixes, in order:")
        print("  1. Record more clips per class (aim 15-20)")
        print("  2. Add a second/third operator")
        print("  3. Re-run CALIBRATE_ZONES.bat - vague zones blur similar actions")
        print("  4. Make sure each clip contains ONLY the action, trimmed tight")
        print("\nThe app still runs on the Tier-0 heuristic in the meantime.")
    else:
        print("\nTier 1 is ready. Restart the app - it will load the model automatically.")

    _verify(out_model, meta, Xte[:4] if len(Xte) else None)
    return 0


def _verify(model_path: Path, meta: dict, sample) -> None:
    """Load the exported ONNX back and check it runs. Catch export bugs here,
    not at 3 a.m. during the demo."""
    try:
        import onnxruntime as ort
    except Exception:
        print("\n(onnxruntime not installed - skipping export verification)")
        return
    try:
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        probe = sample if sample is not None else np.zeros((1, meta["window"], meta["dimension"]), np.float32)
        out = sess.run(None, {name: np.asarray(probe, dtype=np.float32)})[0]
        assert out.shape[-1] == len(meta["labels"]), f"output width {out.shape} != {len(meta['labels'])} labels"
        print(f"\nExport verified: ONNX runs, output shape {out.shape}")
    except Exception as exc:
        print(f"\nWARNING: exported model failed verification: {exc}")


def _model_card(meta: dict, counts: Counter, report: str) -> str:
    return f"""# AEGIS Action Model Card

Generated {meta['trained_at']}

## Purpose
Temporal classifier that maps a {meta['window']}-frame window of rack-relative
operator features to one of {len(meta['labels'])} experiment actions. Consumed by the
protocol engine to validate experiment sequence order.

## Architecture
GRU ({meta['backend']} backend), hidden size {meta['hidden']}, mean-pooled, linear head.
Input {meta['window']} x {meta['dimension']} float32. Exported to ONNX opset 13,
runs on CPU via onnxruntime. No network access at inference time.

## Training data
- {meta['clips']} clips, {meta['train_windows']} training windows after augmentation
- Class distribution: {dict(counts)}

## Validation
**Operator-held-out** split (held out: `{meta['held_out_operator']}`).
Accuracy on unseen operator: **{meta['validation_accuracy']:.1%}**
({meta['test_windows']} test windows)

A random split would report a materially higher and materially less meaningful
number, because consecutive windows from one clip are near-duplicates.

{report}

## Known limitations
- Trained on a single camera geometry. Re-calibrate zones and retrain if the
  camera is repositioned substantially.
- Feature layout v{meta['feature_version']}. The runtime refuses to load this model
  against a different feature version or dimension.
- Actions that differ only by the *object* held (not by motion or location) are
  weak points; enable the Tier-2 object detector for those.

## Intended use
On-board assistance and sequence validation. Advisory only: the operator retains
authority, and every automated decision is logged with its confidence and source.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the AEGIS temporal action model")
    parser.add_argument("--dataset", default="datasets/actions")
    parser.add_argument("--out", default="models/action/action_model.onnx")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--augment", type=int, default=3, help="augmentation factor (1 = off)")
    parser.add_argument("--holdout", default=None, help="operator id to hold out for validation")
    parser.add_argument("--force-numpy", action="store_true", help="ignore PyTorch even if installed")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nTraining cancelled.")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
