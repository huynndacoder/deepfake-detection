import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import (
    discover_models,
    load_model_from_checkpoint,
    get_gradcam_target,
    get_model_display_name,
)
from src.predict import preprocess_image, predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEMO_DIR = PROJECT_ROOT / "assets" / "demo"


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
    return overlay


def create_color_legend(width=200, height=20):
    import cv2

    gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    legend = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    return legend


def make_heatmap_only(cam, target_shape):
    import cv2

    h, w = target_shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap


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


def run_analysis(uploaded_image, model_path):
    model, model_type, device = load_model_from_checkpoint(model_path)
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
        "model_type": model_type,
        "heatmap_overlay": heatmap_overlay,
        "heatmap_only": heatmap_only,
        "original_rgb": original_rgb,
        "stats": stats,
        "caption": caption,
    }


def load_demo_images():
    if not DEMO_DIR.exists():
        return []
    images = sorted(DEMO_DIR.glob("*"))
    return [
        {"name": f.stem.replace("_", " ").title(), "path": str(f)}
        for f in images
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ]


def render_result(confidence_threshold):
    res = st.session_state.result
    if res is None:
        return

    is_low_confidence = res["confidence"] < confidence_threshold
    color = "red" if res["label"] == "fake" else "green"
    emoji = "🤖" if res["label"] == "fake" else "👤"

    with st.container(border=True):
        col_head, col_conf = st.columns([2, 1])
        with col_head:
            st.markdown(
                f"<h2 style='color:{color}; margin:0;'>{emoji} {res['label'].upper()}</h2>",
                unsafe_allow_html=True,
            )
        with col_conf:
            st.metric("Confidence", f"{res['confidence']:.1%}")

        st.progress(
            float(res["confidence"]),
            text=f"Confidence — {res['confidence']:.1%}",
        )

        class_names = ["Fake", "Real"]
        prob_df = {
            "Class": class_names,
            "Probability": [res["probs"][0], res["probs"][1]],
        }
        st.bar_chart(prob_df, x="Class", y="Probability", horizontal=True)

        if is_low_confidence:
            st.warning(
                f"Low-confidence prediction — confidence is below the "
                f"selected threshold ({confidence_threshold:.0%}). "
                f"This result should be interpreted with caution."
            )
        else:
            confidence_level = "High" if res["confidence"] >= 0.90 else "Moderate"
            st.success(
                f"{confidence_level}-confidence prediction. "
                f"The model is relatively certain about this result."
            )

    st.caption(f"Prediction made using {res.get('display_name', res['model_type'])}")


def render_gradcam_tab():
    res = st.session_state.result
    if res is None:
        return

    try:
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.image(
                res["original_rgb"],
                caption="Preprocessed image (224x224)",
                use_container_width=True,
            )
        with viz_col2:
            st.image(
                res["heatmap_only"],
                caption="Grad-CAM heatmap",
                use_container_width=True,
            )

        st.image(
            res["heatmap_overlay"],
            caption="Overlay — areas the model relied on for this prediction",
            use_container_width=True,
        )

        legend = create_color_legend()
        st.image(legend, use_container_width=True)
        leg_col1, leg_col2 = st.columns([0.5, 0.5])
        with leg_col1:
            st.caption("Low influence")
        with leg_col2:
            st.caption("High influence")

    except Exception as e:
        st.warning(f"Grad-CAM visualization failed: {e}")


def render_stats_tab():
    res = st.session_state.result
    if res is None:
        return

    stats = res["stats"]
    st1, st2, st3 = st.columns(3)
    with st1:
        st.metric(
            "Active Area",
            f"{stats['active_ratio']:.0%}",
            help="Fraction of the image with activation above 0.20",
        )
    with st2:
        st.metric(
            "Peak Activation",
            f"{stats['max_activation']:.0%}",
            help="Maximum activation value in the heatmap",
        )
    with st3:
        n = stats["n_regions"]
        st.metric(
            "Focus Regions",
            str(n) if n > 0 else "1",
            help="Number of distinct high-activation areas",
        )

    st.info(res["caption"])


st.set_page_config(page_title="Deepfake Detector", page_icon="🔍", layout="wide")

st.title("Deepfake Face Detector")
st.markdown("Upload a face image to detect whether it is **real** or **AI-generated**.")

available_models = discover_models(MODELS_DIR)

if not available_models:
    st.error(
        "No trained models found in `models/`. Train one first:\n\n"
        "```bash\npython -m src.run_training --model mobilenet\n```"
    )
    st.stop()

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "result" not in st.session_state:
    st.session_state.result = None
if "selected_model_path" not in st.session_state:
    st.session_state.selected_model_path = available_models[0]["path"]

with st.expander("How to use", icon="📖"):
    st.markdown(
        """
    1. **Select a model** in the sidebar — choose which trained detector to use.
    2. **Upload an image** below, or pick a demo image if available.
    3. **Click "Analyze Image"** — the model will process it and show results.
    4. **Review the tabs** — inspect the prediction, Grad-CAM heatmap, and activation statistics.
    """
    )

with st.sidebar:
    st.header("Model")

    selected_model = next(
        (
            m
            for m in available_models
            if m["path"] == st.session_state.selected_model_path
        ),
        available_models[0],
    )

    model_labels = [
        m["display_name"] + " (" + m["name"] + ")" for m in available_models
    ]
    default_idx = next(
        (
            i
            for i, m in enumerate(available_models)
            if m["path"] == st.session_state.selected_model_path
        ),
        0,
    )

    selected_label = st.selectbox(
        "Select Model",
        options=model_labels,
        index=default_idx,
        help="Choose which trained model to use for detection.",
    )
    chosen_model = available_models[model_labels.index(selected_label)]

    if chosen_model["path"] != st.session_state.selected_model_path:
        st.session_state.selected_model_path = chosen_model["path"]
        st.session_state.result = None
        st.rerun()

    st.caption(f"Checkpoint: `{chosen_model['name']}.pt`")
    st.caption(f"Architecture: {get_model_display_name(chosen_model['model_type'])}")

    st.divider()
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
        f"""
    **Dataset:** 10,000 real · 10,000 fake
    faces at 256×256 resolution.
    
    **Classes:** Real · Fake
    """
    )
    st.divider()
    st.markdown("*DS102 — Statistical Machine Learning*")

demo_images = load_demo_images()
if demo_images:
    st.markdown("#### Demo Images")
    COLS_PER_ROW = 4
    rows = (len(demo_images) + COLS_PER_ROW - 1) // COLS_PER_ROW
    for row in range(rows):
        cols = st.columns(COLS_PER_ROW)
        for ci in range(COLS_PER_ROW):
            idx = row * COLS_PER_ROW + ci
            if idx >= len(demo_images):
                break
            demo = demo_images[idx]
            with cols[ci]:
                try:
                    demo_pil = Image.open(demo["path"])
                    st.image(demo_pil, caption=demo["name"], use_container_width=True)
                except Exception:
                    st.image(
                        np.zeros((100, 100, 3), dtype=np.uint8),
                        caption=demo["name"],
                        use_container_width=True,
                    )
                if st.button("Use", key=f"demo_{idx}", use_container_width=True):
                    st.session_state.uploaded_image = Image.open(demo["path"]).copy()
                    st.session_state.result = None
    st.divider()

uploaded_file = st.file_uploader(
    "Choose a face image...",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Upload a face image to check if it's real or AI-generated.",
)

if uploaded_file is not None:
    st.session_state.uploaded_image = Image.open(uploaded_file)
    st.session_state.result = None

if st.session_state.uploaded_image is not None:
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.image(
            st.session_state.uploaded_image,
            caption="Uploaded Image",
            use_container_width=True,
        )

    if st.button("Analyze Image", type="primary", use_container_width=True):
        with st.spinner(f"Loading {chosen_model['display_name']}..."):
            try:
                result = run_analysis(
                    st.session_state.uploaded_image,
                    st.session_state.selected_model_path,
                )
                result["display_name"] = chosen_model["display_name"]
                st.session_state.result = result
            except FileNotFoundError:
                st.error("Model file not found. Please train the model first.")
                st.stop()

    if st.session_state.result is not None:
        with col_right:
            tab_pred, tab_gradcam, tab_stats = st.tabs(
                ["Prediction", "Grad-CAM", "Stats"]
            )
            with tab_pred:
                render_result(confidence_threshold)
            with tab_gradcam:
                render_gradcam_tab()
            with tab_stats:
                render_stats_tab()

st.divider()
st.caption(
    "This tool provides model-assisted predictions and should not be used "
    "as definitive evidence. All results are statistical estimates "
    "based on the training data and model architecture."
)
