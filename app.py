import streamlit as st
import cv2
import numpy as np
import time

from preprocess import load_image
from perturbation import generate_adversarial
from evaluation import get_embedding, cosine_similarity, compute_ssim
from face_detection import detect_face


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI-Resistant Image Protection",
    layout="centered"
)

st.title("AI-Resistant Image Protection System")

st.write(
"""
Upload an image to generate a **protected version** that resists AI training
while remaining visually similar to humans.
"""
)

# -----------------------------
# PERTURBATION STRENGTH
# -----------------------------

st.subheader("Protection Strength")

epsilon_step = st.slider(
    "Adjust perturbation strength",
    min_value=1,
    max_value=8,
    value=4
)

epsilon = epsilon_step * 0.005

st.write(f"Selected epsilon value: **{epsilon:.3f}**")


if epsilon <= 0.01:
    st.info("Low protection — minimal perturbation")
elif epsilon <= 0.025:
    st.info("Balanced protection")
else:
    st.warning("High protection — stronger perturbation")


# -----------------------------
# IMAGE UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, 1)

    display_image = image.copy()


    # -----------------------------
    # FACE DETECTION
    # -----------------------------

    face_box = detect_face(image)

    if face_box is not None:

        x1, y1, x2, y2 = face_box

        cv2.rectangle(display_image, (x1,y1), (x2,y2), (0,255,0), 3)


    st.subheader("Original Image")

    st.image(
        cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB),
        width=500
    )


    # -----------------------------
    # GENERATE PERTURBATION
    # -----------------------------

    start_time = time.time()

    with st.spinner("Generating AI-resistant image..."):

        protected = generate_adversarial(
            image,
            epsilon=epsilon,
            steps=10
        )

    runtime = time.time() - start_time


    st.subheader("Protected Image")

    st.image(
        cv2.cvtColor(protected, cv2.COLOR_BGR2RGB),
        width=500
    )


    # -----------------------------
    # METRICS
    # -----------------------------

    emb_original = get_embedding(image)
    emb_protected = get_embedding(protected)

    similarity = cosine_similarity(
        emb_original,
        emb_protected
    ).item()

    ssim_score = compute_ssim(image, protected)

    noise = protected.astype(np.float32) - image.astype(np.float32)
    noise_mag = np.mean(np.abs(noise))


    st.subheader("Protection Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Embedding Similarity", f"{similarity:.3f}")
    col2.metric("SSIM", f"{ssim_score:.3f}")
    col3.metric("Noise Magnitude", f"{noise_mag:.3f}")
    col4.metric("Processing Time", f"{runtime:.2f}s")


    # -----------------------------
    # HEATMAP
    # -----------------------------

    heatmap = np.abs(noise)

    heatmap = heatmap / (heatmap.max() + 1e-8)

    heatmap = (heatmap * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


    st.subheader("Perturbation Heatmap")

    st.image(
        cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
        width=500
    )


    # -----------------------------
    # COMPARISON VIEW
    # -----------------------------

    st.subheader("Original vs Protected")

    comparison = np.hstack((image, protected))

    st.image(
        cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB),
        caption="Left: Original | Right: Protected",
        width=800
    )


    # -----------------------------
    # DOWNLOAD
    # -----------------------------

    _, buffer = cv2.imencode(".png", protected)

    st.download_button(
        label="Download Protected Image",
        data=buffer.tobytes(),
        file_name="protected_image.png",
        mime="image/png"
    )