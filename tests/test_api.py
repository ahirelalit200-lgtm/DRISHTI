"""Unit tests for FastAPI endpoint src/aegis/api.py."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aegis.api import app


def test_health_check():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "VIKRAM 1 AI-HAR API" in data["service"]


def test_websocket_stream():
    client = TestClient(app)
    # Create a small dummy 100x100 black image
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", dummy_img)
    base64_str = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("utf-8")

    with client.websocket_connect("/ws/stream") as websocket:
        websocket.send_json({"image": base64_str})
        data = websocket.receive_json()
        assert "next_instruction" in data
        assert "action" in data
        assert "status" in data
