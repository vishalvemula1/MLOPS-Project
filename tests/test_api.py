"""
Integration tests for FastAPI endpoints (app/main.py).
Uses FastAPI's TestClient - no real server is started.
Model inference is stubbed so no GPU / weights are needed.
"""
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Skip entire module if torch is not installed
torch = pytest.importorskip("torch", reason="torch not installed")
import torch.nn as nn


# ---------------------------------------------------------------------------
# Stub model & fixtures
# ---------------------------------------------------------------------------
class _BenignStub(nn.Module):
    """Always predicts benign."""
    def forward(self, x):
        return torch.tensor([[5.0, -5.0]])


class _MalignantStub(nn.Module):
    """Always predicts malignant."""
    def forward(self, x):
        return torch.tensor([[-5.0, 5.0]])


def _make_image_bytes(mode="RGB", size=(100, 100)) -> bytes:
    buf = io.BytesIO()
    fmt = "PNG" if mode == "RGBA" else "JPEG"
    Image.new(mode, size, color=(100, 150, 200)).save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


@pytest.fixture()
def client_benign(monkeypatch):
    """TestClient with model stubbed to always predict benign."""
    import app.inference as inf
    inf.reset_model()
    monkeypatch.setattr(inf, "load_model", lambda *a, **kw: _BenignStub())
    monkeypatch.setattr(inf, "_model", _BenignStub())

    from app.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    inf.reset_model()


@pytest.fixture()
def client_malignant(monkeypatch):
    """TestClient with model stubbed to always predict malignant."""
    import app.inference as inf
    inf.reset_model()
    monkeypatch.setattr(inf, "load_model", lambda *a, **kw: _MalignantStub())
    monkeypatch.setattr(inf, "_model", _MalignantStub())

    from app.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    inf.reset_model()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_status_ok(self, client_benign):
        resp = client_benign.get("/health")
        assert resp.status_code == 200

    def test_returns_ok(self, client_benign):
        data = client_benign.get("/health").json()
        assert data["status"] == "ok"

    def test_returns_classes(self, client_benign):
        data = client_benign.get("/health").json()
        assert "classes" in data
        assert set(data["classes"]) == {"benign", "malignant"}


# ---------------------------------------------------------------------------
# / (HTML UI)
# ---------------------------------------------------------------------------
class TestIndexUI:
    def test_returns_html(self, client_benign):
        resp = client_benign.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_contains_title(self, client_benign):
        body = client_benign.get("/").text
        assert "Skin Cancer Detector" in body


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------
class TestPredict:
    def test_valid_image_benign(self, client_benign):
        resp = client_benign.post(
            "/predict",
            files={"file": ("test.jpg", _make_image_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_class"] == "benign"

    def test_valid_image_malignant(self, client_malignant):
        resp = client_malignant.post(
            "/predict",
            files={"file": ("test.jpg", _make_image_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_class"] == "malignant"

    def test_response_schema(self, client_benign):
        resp = client_benign.post(
            "/predict",
            files={"file": ("test.jpg", _make_image_bytes(), "image/jpeg")},
        )
        data = resp.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "benign" in data["probabilities"]
        assert "malignant" in data["probabilities"]

    def test_confidence_is_float(self, client_benign):
        resp = client_benign.post(
            "/predict",
            files={"file": ("test.jpg", _make_image_bytes(), "image/jpeg")},
        )
        data = resp.json()
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self, client_benign):
        resp = client_benign.post(
            "/predict",
            files={"file": ("test.jpg", _make_image_bytes(), "image/jpeg")},
        )
        probs = resp.json()["probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-2

    def test_empty_file_rejected(self, client_benign):
        resp = client_benign.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_png_accepted(self, client_benign):
        buf = io.BytesIO()
        Image.new("RGB", (50, 50), color=(200, 100, 50)).save(buf, format="PNG")
        buf.seek(0)
        resp = client_benign.post(
            "/predict",
            files={"file": ("test.png", buf.read(), "image/png")},
        )
        assert resp.status_code == 200

    def test_no_file_returns_422(self, client_benign):
        resp = client_benign.post("/predict")
        assert resp.status_code == 422

    def test_non_image_content_type(self, client_benign):
        resp = client_benign.post(
            "/predict",
            files={"file": ("data.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 415

    def test_rgba_image_accepted(self, client_benign):
        """RGBA PNG should be accepted and silently converted to RGB."""
        resp = client_benign.post(
            "/predict",
            files={"file": ("rgba.png", _make_image_bytes(mode="RGBA"), "image/png")},
        )
        assert resp.status_code == 200
