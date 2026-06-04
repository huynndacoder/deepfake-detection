import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import build_model, get_registered_model_names
from src.dataset import create_dataloaders
from src.train import train_model
from src.evaluate import evaluate_model, print_metrics


def main():
    parser = argparse.ArgumentParser(description="Train a deepfake detection model.")
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet",
        choices=get_registered_model_names(),
        help="Model architecture to train.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    model_type = args.model
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    MODEL_SAVE_PATH = PROJECT_ROOT / "models" / f"{model_type}_best.pt"
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    NUM_WORKERS = args.workers

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        data_dir=DATA_DIR,
        image_size=224,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    print(f"Classes: {class_to_idx}")
    print(
        f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}"
    )

    print(f"\nBuilding {model_type}...")
    model = build_model(model_type)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nTraining...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        save_path=str(MODEL_SAVE_PATH),
        patience=5,
        model_type=model_type,
    )

    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, test_loader)
    print_metrics(metrics)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
