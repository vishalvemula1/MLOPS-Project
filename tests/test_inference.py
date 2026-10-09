"""
Tests for the inference module (app/inference.py).
These tests run without any real model weights - they patch / build
a tiny mock model so the CI pipeline stays fast and dependency-light.
"""
import io
import pytest
from PIL import Image

# Skip entire module if torch is not installed
torch = pytest.importorskip("torch", reason="torch not installed")
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_dummy_image_bytes(mode: str = "RGB", size: tuple = (300, 300)) -> bytes:
    """Return raw bytes for a small synthetic image."""
    buf = io.BytesIO()
    fmt = "PNG" if mode == "RGBA" else "JPEG"
    Image.new(mode, size, color=(120, 80, 60)).save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


def _make_tiny_model(num_classes: int = 2) -> nn.Module:
    """Return a tiny 2-class Linear model that mimics the real model interface."""
    model = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_classes))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def reset_cached_model():
    """Ensure each test starts with a clean singleton state."""
    from app import inference
    inference.reset_model()
    yield
    inference.reset_model()


# ---------------------------------------------------------------------------
# inference.py unit tests
# ---------------------------------------------------------------------------
class TestBuildModel:
    def test_output_classes(self):
        """build_model should produce a model with exactly 2 output neurons."""
        from app.inference import build_model
        model = build_model(num_classes=2)
        assert model.fc.out_features == 2

    def test_custom_num_classes(self):
        from app.inference import build_model
        model = build_model(num_classes=7)
        assert model.fc.out_features == 7


class TestLoadModel:
    def test_returns_nn_module(self):
        """load_model should always return an nn.Module in eval mode."""
        from app.inference import load_model
        model = load_model(weights_path=None)
        assert isinstance(model, nn.Module)
        assert not model.training   # must be in eval mode

    def test_singleton(self):
        """Calling load_model twice must return the same object."""
        from app.inference import load_model
        m1 = load_model()
        m2 = load_model()
        assert m1 is m2

    def test_missing_weights_does_not_crash(self, tmp_path):
        """If the weights file doesn't exist, fall back gracefully."""
        from app.inference import load_model
        model = load_model(weights_path=str(tmp_path / "does_not_exist.pth"))
        assert isinstance(model, nn.Module)


class TestPredictFromBytes:
    def _patch_load_model(self, monkeypatch, model):
        """Replace load_model with a stub returning *model*."""
        import app.inference as inf
        monkeypatch.setattr(inf, "load_model", lambda *a, **kw: model)
        monkeypatch.setattr(inf, "_model", model)

    def test_returns_expected_keys(self, monkeypatch):
        from app.inference import predict_from_bytes

        # Build a small stub model
        class _Stub(nn.Module):
            def forward(self, x):
                return torch.tensor([[0.3, 0.7]])

        self._patch_load_model(monkeypatch, _Stub())

        result = predict_from_bytes(_make_dummy_image_bytes())
        assert "predicted_class" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_predicted_class_is_valid(self, monkeypatch):
        from app.inference import predict_from_bytes, CLASS_NAMES

        class _Stub(nn.Module):
            def forward(self, x):
                return torch.tensor([[0.8, 0.2]])

        self._patch_load_model(monkeypatch, _Stub())

        result = predict_from_bytes(_make_dummy_image_bytes())
        assert result["predicted_class"] in CLASS_NAMES

    def test_confidence_in_range(self, monkeypatch):
        from app.inference import predict_from_bytes

        class _Stub(nn.Module):
            def forward(self, x):
                return torch.tensor([[2.0, -1.0]])  # large logit gap

        self._patch_load_model(monkeypatch, _Stub())

        result = predict_from_bytes(_make_dummy_image_bytes())
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self, monkeypatch):
        from app.inference import predict_from_bytes

        class _Stub(nn.Module):
            def forward(self, x):
                return torch.tensor([[1.5, 0.5]])

        self._patch_load_model(monkeypatch, _Stub())

        result = predict_from_bytes(_make_dummy_image_bytes())
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-3

    def test_malignant_prediction(self, monkeypatch):
        from app.inference import predict_from_bytes

        class _Stub(nn.Module):
            def forward(self, x):
                return torch.tensor([[-5.0, 5.0]])  # strongly malignant

        self._patch_load_model(monkeypatch, _Stub())

        result = predict_from_bytes(_make_dummy_image_bytes())
        assert result["predicted_class"] == "malignant"
        assert result["confidence"] > 0.99

    def test_benign_prediction(self, monkeypatch):
        from app.inference import predict_from_bytes

        class _Stub(nn.Module):
            def forward(self, x):
                return torch.tensor([[5.0, -5.0]])  # strongly benign

        self._patch_load_model(monkeypatch, _Stub())

        result = predict_from_bytes(_make_dummy_image_bytes())
        assert result["predicted_class"] == "benign"
        assert result["confidence"] > 0.99

    def test_rgba_image_converted(self, monkeypatch):
        """RGBA images should be silently converted to RGB."""
        from app.inference import predict_from_bytes

        class _Stub(nn.Module):
            def forward(self, x):
                return torch.tensor([[0.6, 0.4]])

        self._patch_load_model(monkeypatch, _Stub())

        result = predict_from_bytes(_make_dummy_image_bytes(mode="RGBA"))
        assert result["predicted_class"] in ["benign", "malignant"]

    def test_small_image_handled(self, monkeypatch):
        """Very small images should be resized and not raise an error."""
        from app.inference import predict_from_bytes

        class _Stub(nn.Module):
            def forward(self, x):
                return torch.tensor([[0.5, 0.5]])

        self._patch_load_model(monkeypatch, _Stub())

        result = predict_from_bytes(_make_dummy_image_bytes(size=(10, 10)))
        assert "predicted_class" in result
