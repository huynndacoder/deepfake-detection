import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models


MODEL_REGISTRY = {}


def register_model(name, builder, gradcam_target_fn, display_name, image_size=224):
    MODEL_REGISTRY[name] = {
        "builder": builder,
        "gradcam_target": gradcam_target_fn,
        "display_name": display_name,
        "image_size": image_size,
    }


def build_model(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]["builder"]()


def get_gradcam_target(model, name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]["gradcam_target"](model)


def get_model_display_name(name):
    if name not in MODEL_REGISTRY:
        return name
    return MODEL_REGISTRY[name]["display_name"]


def get_registered_model_names():
    return list(MODEL_REGISTRY.keys())


# ── MobileNet ────────────────────────────────────────────


def _build_mobilenet():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    return model


def _mobilenet_gradcam_target(model):
    return model.features[-1]


register_model("mobilenet", _build_mobilenet, _mobilenet_gradcam_target, "MobileNet")


# ── EfficientNet-B0 ──────────────────────────────────────


def _build_efficientnetb0():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model


def _efficientnetb0_gradcam_target(model):
    return model.features[-1]


register_model(
    "efficientnetb0",
    _build_efficientnetb0,
    _efficientnetb0_gradcam_target,
    "EfficientNet-B0",
)


# ── Custom CNN ───────────────────────────────────────────


class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _build_customcnn():
    return CustomCNN(num_classes=2)


def _customcnn_gradcam_target(model):
    return model.features[-1]


register_model("customcnn", _build_customcnn, _customcnn_gradcam_target, "Custom CNN")


# ── Checkpoint utilities ─────────────────────────────────


def load_model_from_checkpoint(checkpoint_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "model_type" in checkpoint:
        model_type = checkpoint["model_type"]
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        model_type = "mobilenet"
        state_dict = checkpoint
    elif isinstance(checkpoint, dict):
        raise ValueError(
            f"Cannot determine model type from checkpoint: {checkpoint_path}"
        )

    model = build_model(model_type)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, model_type, device


def save_checkpoint(model, save_path, model_type):
    checkpoint = {
        "model_type": model_type,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, save_path)


def discover_models(models_dir):
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []

    filename_map = {
        "mobilenet": "mobilenet",
        "efficientnetb0": "efficientnetb0",
        "customcnn": "customcnn",
    }

    available = []
    for pt_file in sorted(models_dir.glob("*.pt")):
        name = pt_file.stem
        model_type = filename_map.get(name.split("_")[0])

        if model_type is None:
            try:
                checkpoint = torch.load(pt_file, map_location="cpu", weights_only=True)
                if isinstance(checkpoint, dict) and "model_type" in checkpoint:
                    model_type = checkpoint["model_type"]
                else:
                    model_type = "mobilenet"
            except Exception:
                continue

        available.append(
            {
                "path": str(pt_file),
                "name": name,
                "model_type": model_type,
                "display_name": get_model_display_name(model_type),
            }
        )

    return available
