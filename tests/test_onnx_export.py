"""The exported ONNX model must compute exactly what the trainer trained.

This is the test that caught a real bug: ONNX's GRU op defines the update gate
as ``Ht = (1-z)*h~ + z*H(t-1)`` while the NumPy implementation uses the opposite
assignment. Without the compensating sign flip in the exporter, training reported
100% and the deployed model quietly produced different logits.

Any change to ``NumpyGRU.forward`` or ``export_numpy_onnx`` must keep this green.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from aegis.tools.train_action_model import NumpyGRU, export_numpy_onnx, softmax


def _randomised_model(dim, hidden, classes, seed=4):
    rng = np.random.default_rng(seed)
    model = NumpyGRU(dim, hidden, classes, seed=seed)
    # Non-zero biases matter: zeros can mask a gate-ordering mistake.
    for key in ("bz", "br", "bh", "bo"):
        model.p[key] = rng.normal(0.0, 0.5, model.p[key].shape).astype(np.float32)
    return model


@pytest.mark.parametrize(
    "dim,window,classes,hidden,seed",
    [(17, 9, 5, 13, 4), (62, 32, 4, 48, 11), (8, 4, 2, 6, 21)],
)
def test_onnx_export_matches_numpy_forward(dim, window, classes, hidden, seed):
    model = _randomised_model(dim, hidden, classes, seed)
    rng = np.random.default_rng(seed + 100)
    X = rng.normal(0.0, 1.0, (6, window, dim)).astype(np.float32)

    reference, _ = model.forward(X)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.onnx"
        export_numpy_onnx(model, path, window, dim, classes)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        exported = session.run(None, {session.get_inputs()[0].name: X})[0]

    assert exported.shape == reference.shape
    max_diff = float(np.abs(reference - exported).max())
    assert max_diff < 1e-4, f"logit mismatch of {max_diff:.2e} between NumPy and ONNX"
    assert (reference.argmax(1) == exported.argmax(1)).all()

    # probabilities, not just logits, must agree -- that is what the runtime uses
    prob_diff = float(np.abs(softmax(reference) - softmax(exported)).max())
    assert prob_diff < 1e-4


def test_exported_model_accepts_a_dynamic_batch():
    model = _randomised_model(12, 16, 3, 10)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.onnx"
        export_numpy_onnx(model, path, 7, 12, 3)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        for batch in (1, 3, 9):
            out = session.run(None, {name: np.zeros((batch, 7, 12), np.float32)})[0]
            assert out.shape == (batch, 3)


def test_learned_recognizer_loads_an_exported_model():
    """The runtime's loader must accept what the trainer produces."""
    import json

    from aegis.perception.recognizer import LearnedRecognizer

    dim, window, classes = 20, 8, 3
    model = _randomised_model(dim, 12, classes)
    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "action_model.onnx"
        meta_path = Path(tmp) / "action_model.meta.json"
        export_numpy_onnx(model, model_path, window, dim, classes)
        meta_path.write_text(
            json.dumps(
                {
                    "labels": ["grasp_item", "insert_item", "idle"],
                    "window": window,
                    "dimension": dim,
                    "feature_version": 2,
                    "mean": [0.0] * dim,
                    "std": [1.0] * dim,
                }
            ),
            encoding="utf-8",
        )

        recogniser = LearnedRecognizer(model_path, meta_path)
        assert recogniser.available, recogniser.error
        assert recogniser.compatible_with(dim, 2)
        assert not recogniser.compatible_with(dim + 1, 2), "dimension mismatch must be rejected"
        assert not recogniser.compatible_with(dim, 99), "feature version mismatch must be rejected"

        result = recogniser.predict(np.zeros((window, dim), np.float32))
        assert result.source == "learned"
        assert 0.0 <= result.confidence <= 1.0
        # 'idle' maps to a None action so the protocol engine ignores it
        assert result.action in (None, "grasp_item", "insert_item")
