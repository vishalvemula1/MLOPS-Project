"""
Skin Cancer Inference Module
Wraps the ResNet-50 model for binary (benign / malignant) prediction.
Designed to work both with a saved .pth checkpoint and in mock/test mode
when no checkpoint is present (e.g., in CI).
"""
import io
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES = ["benign", "malignant"]
INPUT_SIZE = (300, 225)   # (width, height) – matches training config

# Where the app looks for a saved checkpoint (can be overridden via env var)
DEFAULT_WEIGHTS_PATH = os.getenv(
    "MODEL_WEIGHTS_PATH",
    str(Path(__file__).parent.parent / "models" / "model_weights.pth"),
)

# ── transforms ─────────────────────────────────────────────────────────────────
_inference_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE[::-1]),   # PIL Resize takes (height, width)
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ── model builder ──────────────────────────────────────────────────────────────
def build_model(num_classes: int = 2) -> nn.Module:
    """Return a ResNet-50 with the classification head replaced."""
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ── singleton loader ───────────────────────────────────────────────────────────
_model: Optional[nn.Module] = None
_device: Optional[torch.device] = None


def load_model(weights_path: Optional[str] = None) -> nn.Module:
    """
    Load (or return cached) the inference model.

    If *weights_path* is provided, load from that checkpoint.
    If the default path does not exist either, fall back to an
    untrained model so the API still starts (useful in CI / demo).
    """
    global _model, _device

    if _model is not None:
        return _model

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", _device)

    model = build_model(num_classes=2)

    path = weights_path or DEFAULT_WEIGHTS_PATH
    if path and Path(path).exists():
        logger.info("Loading weights from %s", path)
        try:
            state = torch.load(path, map_location=_device, weights_only=False)
            # Handle both raw state_dicts and checkpoint dicts
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            if isinstance(state, dict):
                # If checkpoint has a classification head with different number of classes (e.g. 1000)
                # perform transfer learning by loading all matching feature layers as trained in model.py
                fc_weight = state.get("fc.weight")
                if fc_weight is not None and fc_weight.shape != model.fc.weight.shape:
                    logger.info("Transferring pretrained feature backbone weights from %s", path)
                    filtered_state = {k: v for k, v in state.items() if "fc" not in k}
                    model_dict = model.state_dict()
                    model_dict.update(filtered_state)
                    model.load_state_dict(model_dict)
                else:
                    model.load_state_dict(state)
                logger.info("Weights loaded successfully.")
            else:
                logger.warning("Unsupported state type (%s) -- using untrained model.", type(state))
        except Exception as exc:
            logger.warning("Weight load failed (%s) -- using untrained model.", exc)
    else:
        logger.warning(
            "No weights found at '%s'. Using untrained model (predictions are random).",
            path,
        )

    model.to(_device)
    model.eval()
    _model = model
    return _model


def reset_model() -> None:
    """Clear the cached model (useful in tests)."""
    global _model, _device
    _model = None
    _device = None


# ── prediction ─────────────────────────────────────────────────────────────────
def predict_from_bytes(image_bytes: bytes) -> dict:
    """
    Run inference on raw image bytes.

    Returns
    -------
    dict with keys:
        predicted_class : str   - "benign" or "malignant"
        confidence      : float - probability of the predicted class (0-1)
        probabilities   : dict  - {class_name: probability} for all classes
    """
    model = load_model()

    # --- preprocess --------------------------------------------------------
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _inference_transform(image).unsqueeze(0)   # (1, C, H, W)

    try:
        device = next(model.parameters()).device
    except (StopIteration, TypeError):
        device = _device or torch.device("cpu")
    tensor = tensor.to(device)

    # --- forward pass ------------------------------------------------------
    with torch.no_grad():
        logits = model(tensor)                           # (1, num_classes)
        probs = torch.softmax(logits, dim=1).squeeze()   # (num_classes,)

    probs_list = probs.cpu().tolist()
    predicted_idx = int(torch.argmax(probs).item())

    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": round(probs_list[predicted_idx], 4),
        "probabilities": {
            cls: round(p, 4) for cls, p in zip(CLASS_NAMES, probs_list)
        },
    }
