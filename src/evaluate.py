"""
src/evaluate.py
===============
Shared evaluation utilities — dùng cho mọi model trong project.

Hàm chính
---------
evaluate_model(y_true, y_pred, y_prob, model_name, save_dir)
    → in metrics ra màn hình
    → lưu confusion matrix + ROC curve vào reports/
    → trả về dict kết quả để so sánh sau
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


def evaluate_model(
    y_true,
    y_pred,
    y_prob,
    model_name: str,
    save_dir: str,
) -> dict:
    """
    Tính toán và hiển thị đầy đủ metrics cho một model.

    Parameters
    ----------
    y_true     : array-like, shape (N,)  — nhãn thật (0=fake, 1=real)
    y_pred     : array-like, shape (N,)  — nhãn dự đoán
    y_prob     : array-like, shape (N,)  — xác suất lớp "real" (dùng cho ROC)
    model_name : str  — tên model, dùng làm tiêu đề và tên file
    save_dir   : str  — thư mục lưu ảnh (thường là PROJECT_ROOT/reports/)

    Returns
    -------
    dict với các key: accuracy, precision, recall, f1, roc_auc
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Tính metrics ──────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)

    # ── In ra màn hình ────────────────────────────────────────────────────────
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  KẾT QUẢ: {model_name}")
    print(sep)
    print(f"  Accuracy  : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(sep)

    # ── Vẽ Confusion Matrix ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["fake (0)", "real (1)"])
    disp.plot(cmap="Blues", ax=axes[0], colorbar=False)
    axes[0].set_title(f"{model_name}\nConfusion Matrix")

    # ── Vẽ ROC Curve ─────────────────────────────────────────────────────────
    RocCurveDisplay.from_predictions(
        y_true,
        y_prob,
        name=model_name,
        ax=axes[1],
    )
    axes[1].plot([0, 1], [0, 1], "k--", label="Random (AUC=0.50)")
    axes[1].set_title(f"{model_name}\nROC Curve (AUC = {roc_auc:.4f})")
    axes[1].legend(loc="lower right")

    plt.tight_layout()

    # Lưu file — tên file = tên model (bỏ ký tự đặc biệt)
    safe_name = model_name.lower().replace(" ", "_").replace("+", "_")
    save_path = os.path.join(save_dir, f"{safe_name}_evaluation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Đã lưu biểu đồ → {save_path}\n")

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
    }


def print_metrics(metrics):
    print(f"\n{'=' * 52}")
    print(f"  KẾT QUẢ: {metrics.get('model', 'Model')}")
    print(f"{'=' * 52}")
    print(f"  Accuracy  : {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision : {metrics.get('precision', 0):.4f}")
    print(f"  Recall    : {metrics.get('recall', 0):.4f}")
    print(f"  F1-score  : {metrics.get('f1', 0):.4f}")
    print(f"  ROC-AUC   : {metrics.get('roc_auc', 0):.4f}")
    print(f"{'=' * 52}")


def evaluate_model_pt(
    model, loader, model_name="Model", save_dir="reports", device=None
):
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            y_prob.extend(probs[:, 1].cpu().numpy().tolist())
            y_true.extend(labels.numpy().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    return evaluate_model(y_true, y_pred, y_prob, model_name, save_dir)
