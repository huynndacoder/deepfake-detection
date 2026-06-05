import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import load_model_from_checkpoint, get_gradcam_target
from src.predict import preprocess_image, predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "mobilenet_best.pt"


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward(retain_graph=False)

        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(pooled_gradients * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def tensor_to_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor.cpu() * std + mean
    tensor = tensor.clamp(0, 1)
    tensor = tensor.squeeze(0).permute(1, 2, 0).numpy()
    return (tensor * 255).astype(np.uint8)


def overlay_heatmap(image, cam):
    import cv2

    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def make_heatmap_only(cam, target_shape):
    import cv2

    h, w = target_shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def compute_activation_stats(cam):
    import cv2

    active = (cam > 0.20).sum()
    active_ratio = active / cam.size
    high = cam > 0.50
    high_labeled = high.astype(np.uint8)
    n_regions = cv2.connectedComponents(high_labeled)[0] - 1

    return {
        "active_ratio": active_ratio,
        "mean_activation": float(cam.mean()),
        "max_activation": float(cam.max()),
        "n_regions": n_regions,
    }


def generate_interpretive_caption(cam, rgb_image, pred_label, confidence):
    import cv2

    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

        scale_x = cam.shape[1] / rgb_image.shape[1]
        scale_y = cam.shape[0] / rgb_image.shape[0]
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = max(1, int(w * scale_x))
        h = max(1, int(h * scale_y))
        x, y = max(0, x), max(0, y)
        w, h = min(w, cam.shape[1] - x), min(h, cam.shape[0] - y)
        face_cam = cam[y : y + h, x : x + w]

        h_third = max(1, h // 3)
        upper = face_cam[:h_third, :].mean()
        middle = face_cam[h_third : 2 * h_third, :].mean()
        lower = face_cam[2 * h_third :, :].mean()

        regions = [
            ("eye/forehead", upper),
            ("nose/cheek", middle),
            ("mouth/chin", lower),
        ]
        primary_region, primary_score = max(regions, key=lambda r: r[1])

        caption = (
            f"Grad-CAM suggests the model relied most on the "
            f"**{primary_region}** area ({primary_score:.0%} mean activation), "
            f"which contributed to a **{pred_label.upper()}** prediction "
            f"at {confidence:.0%} confidence."
        )
    else:
        h_grid, w_grid = 3, 3
        h_step = max(1, cam.shape[0] // h_grid)
        w_step = max(1, cam.shape[1] // w_grid)
        best_val, best_zone = 0, ""

        for ri in range(h_grid):
            for ci in range(w_grid):
                r0, r1 = ri * h_step, min((ri + 1) * h_step, cam.shape[0])
                c0, c1 = ci * w_step, min((ci + 1) * w_step, cam.shape[1])
                zone_mean = cam[r0:r1, c0:c1].mean()
                if zone_mean > best_val:
                    best_val = zone_mean
                    row_label = "upper" if ri == 0 else "middle" if ri == 1 else "lower"
                    col_label = "left" if ci == 0 else "center" if ci == 1 else "right"
                    best_zone = f"{row_label}-{col_label}"

        caption = (
            f"Grad-CAM suggests the model relied most on the "
            f"**{best_zone}** region ({best_val:.0%} mean activation), "
            f"which contributed to a **{pred_label.upper()}** prediction "
            f"at {confidence:.0%} confidence."
        )

    return caption


def run_analysis(uploaded_image):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model, model_type, device = load_model_from_checkpoint(MODEL_PATH)
    img_tensor = preprocess_image(uploaded_image)
    img_tensor_device = img_tensor.to(device)
    pred_label, confidence, probs = predict(model, img_tensor, device)

    target_layer = get_gradcam_target(model, model_type)
    grad_cam = GradCAM(model, target_layer)
    class_idx = 1 if pred_label == "real" else 0
    cam = grad_cam.generate(img_tensor_device, class_idx)

    original_rgb = tensor_to_image(img_tensor)
    heatmap_overlay = overlay_heatmap(original_rgb, cam)
    heatmap_only = make_heatmap_only(cam, original_rgb.shape)
    stats = compute_activation_stats(cam)
    caption = generate_interpretive_caption(cam, original_rgb, pred_label, confidence)

    return {
        "label": pred_label,
        "confidence": confidence,
        "probs": probs,
        "heatmap_overlay": heatmap_overlay,
        "heatmap_only": heatmap_only,
        "original_rgb": original_rgb,
        "stats": stats,
        "caption": caption,
    }


st.set_page_config(page_title="Deepfake Detector", page_icon="🔍", layout="wide")

st.title("Deepfake Face Detector")
st.markdown(
    "Upload a face image to detect whether it is **real** or **AI-generated** using **MobileNet**."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model not found at `{MODEL_PATH}`. Train it first:\n\n"
        "```bash\npython -m src.run_training --model mobilenet\n```"
    )
    st.stop()

with st.expander("How to use", icon="📖"):
    st.markdown(
        """
    1. **Upload an image** of a face below.
    2. **Click "Analyze Image"** to run detection.
    3. **Review the tabs** — prediction, Grad-CAM heatmap, and activation statistics.
    """
    )

with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Predictions below this threshold are flagged as low-confidence.",
    )

    st.divider()
    st.header("About")
    st.markdown(
        """
    **Model:** MobileNet (Transfer Learning)

    **Dataset:** 10,000 real · 10,000 fake
    faces at 256×256 resolution.

    **Classes:** Real · Fake
    """
    )
    st.divider()
    st.markdown("*DS102 — Statistical Machine Learning*")

uploaded_file = st.file_uploader(
    "Choose a face image...",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Upload a face image to check if it's real or AI-generated.",
)

if "result" not in st.session_state:
    st.session_state.result = None

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.image(uploaded_image, caption="Uploaded Image", width="stretch")

    if st.button("Analyze Image", type="primary", width="stretch"):
        with st.spinner("Running MobileNet..."):
            try:
                st.session_state.result = run_analysis(uploaded_image)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

    if st.session_state.result is not None:
        res = st.session_state.result
        with col_right:
            tab_pred, tab_gradcam, tab_stats = st.tabs(
                ["Prediction", "Grad-CAM", "Stats"]
            )
            with tab_pred:
                color = "red" if res["label"] == "fake" else "green"
                emoji = "🤖" if res["label"] == "fake" else "👤"

                with st.container(border=True):
                    st.markdown(
                        f"<h2 style='color:{color}; margin:0;'>{emoji} {res['label'].upper()}</h2>",
                        unsafe_allow_html=True,
                    )
                    col_m1, col_m2 = st.columns([2, 1])
                    with col_m2:
                        st.metric("Confidence", f"{res['confidence']:.1%}")

                    st.progress(
                        float(res["confidence"]),
                        text=f"Confidence — {res['confidence']:.1%}",
                    )

                    prob_df = {
                        "Class": ["Fake", "Real"],
                        "Probability": [float(res["probs"][0]), float(res["probs"][1])],
                    }
                    st.bar_chart(prob_df, x="Class", y="Probability", horizontal=True)

                    if res["confidence"] < confidence_threshold:
                        st.warning(
                            f"Low-confidence prediction ({confidence_threshold:.0%} threshold). Interpret with caution."
                        )
                    elif res["confidence"] >= 0.90:
                        st.success("High-confidence prediction.")
                    else:
                        st.info("Moderate-confidence prediction.")

                st.caption("Prediction made using MobileNet (Transfer Learning)")

            with tab_gradcam:
                try:
                    viz_col1, viz_col2 = st.columns(2)
                    with viz_col1:
                        st.image(
                            res["original_rgb"],
                            caption="Preprocessed image (224x224)",
                            width="stretch",
                        )
                    with viz_col2:
                        st.image(
                            res["heatmap_only"],
                            caption="Grad-CAM heatmap",
                            width="stretch",
                        )

                    st.image(
                        res["heatmap_overlay"],
                        caption="Overlay — areas the model relied on",
                        width="stretch",
                    )

                    import cv2

                    legend = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (20, 1))
                    legend = cv2.applyColorMap(legend, cv2.COLORMAP_JET)
                    legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
                    st.image(legend, width="stretch")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("Low influence")
                    with c2:
                        st.caption("High influence")

                except Exception as e:
                    st.warning(f"Grad-CAM visualization failed: {e}")

            with tab_stats:
                stats = res["stats"]
                st1, st2, st3 = st.columns(3)
                with st1:
                    st.metric("Active Area", f"{stats['active_ratio']:.0%}")
                with st2:
                    st.metric("Peak Activation", f"{stats['max_activation']:.0%}")
                with st3:
                    n = stats["n_regions"]
                    st.metric("Focus Regions", str(n) if n > 0 else "1")

                st.info(res["caption"])

st.divider()
st.caption(
    "This tool provides model-assisted predictions and should not be used "
    "as definitive evidence. All results are statistical estimates "
    "based on the training data and model architecture."
)
