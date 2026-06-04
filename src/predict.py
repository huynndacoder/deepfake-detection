import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import load_model_from_checkpoint


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(model_path, device=None):
    model, model_type, device = load_model_from_checkpoint(model_path, device)
    return model, device


def preprocess_image(image, image_size=224):
    transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.143)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0)
    return tensor


def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    pred_label = "real" if pred_class.item() == 1 else "fake"
    return pred_label, confidence.item(), probs.cpu().numpy()[0]
