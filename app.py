import streamlit as st
import cv2
import numpy as np
import time

from preprocess import load_image
from perturbation import generate_adversarial
from evaluation import get_embedding, cosine_similarity, compute_ssim, compute_psnr
from face_detection import detect_face

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="AI-Resistant Cloaking",
    page_icon="🛡️",
    layout="wide",  # Upgraded to wide layout for better image comparison
    initial_sidebar_state="expanded"
)

# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
with st.sidebar:
    st.title("🛡️ Protection Settings")
    st.markdown("Adjust the parameters of your adversarial cloak before uploading an image.")

    st.divider()

    st.subheader("Perturbation Strength")
    epsilon_step = st.slider(
        "Select Epsilon Level",
        min_value=1,
        max_value=10,
        value=6,
        help="Higher values provide stronger AI blinding but may introduce slight grain."
    )

    epsilon = epsilon_step * 0.005
    st.write(f"**Current Epsilon: {epsilon:.3f}**")

    # Dynamic status indicator
    if epsilon <= 0.01:
        st.success("🟢 Low Protection (Max Visual Quality)")
    elif epsilon <= 0.025:
        st.info("🔵 Balanced Protection (Recommended)")
    else:
        st.warning("🟠 High Protection (Strong AI Blinding)")

    st.divider()
    st.markdown("### About")
    st.caption(
        "This tool applies an Edge-Aware PGD adversarial attack to disrupt "
        "facial feature extraction by AI models like FaceNet and ArcFace, "
        "while preserving visual fidelity (SSIM) for human viewers."
    )

# -----------------------------
# MAIN DASHBOARD
# -----------------------------
st.title("AI-Resistant Image Cloaking System")
st.markdown(
    "Upload a personal photograph to instantly generate a privacy-preserving version that blinds unauthorized AI data scrapers.")

uploaded_file = st.file_uploader(
    "Drop your image here (JPG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.divider()

    # 1. Read the image (OpenCV loads as BGR)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)

    # 2. CRITICAL FIX: Convert to RGB immediately for the PyTorch backend
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    display_image = image_rgb.copy()

    # Draw face box for visual feedback (using the RGB image)
    face_box = detect_face(image_rgb)
    if face_box is not None:
        x1, y1, x2, y2 = face_box
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 100), 3)

    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.subheader("📷 Original Image")
        # Since display_image is already RGB, we don't need to convert it here
        st.image(display_image, use_container_width=True)

    with col_img2:
        st.subheader("🛡️ Protected Image")

        with st.spinner("Generating adversarial perturbation..."):
            start_time = time.time()
            # 3. Pass the RGB image to the attack function
            protected_rgb = generate_adversarial(image_rgb, epsilon=epsilon, steps=10)
            runtime = time.time() - start_time

        # Display the RGB protected image
        st.image(protected_rgb, use_container_width=True)

        # 4. Convert back to BGR ONLY for downloading
        protected_bgr = cv2.cvtColor(protected_rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".png", protected_bgr)
        st.download_button(
            label="⬇️ Download Protected Image",
            data=buffer.tobytes(),
            file_name="protected_image.png",
            mime="image/png",
            use_container_width=True
        )

    # -----------------------------
    # METRICS DASHBOARD
    # -----------------------------
    st.divider()
    st.subheader("📊 Performance Metrics")

    # 5. Calculate metrics using the RGB images!
    emb_original = get_embedding(image_rgb)
    emb_protected = get_embedding(protected_rgb)
    similarity = cosine_similarity(emb_original, emb_protected).item()

    ssim_score = compute_ssim(image_rgb, protected_rgb)
    psnr_score = compute_psnr(image_rgb, protected_rgb)

    noise = protected_rgb.astype(np.float32) - image_rgb.astype(np.float32)
    noise_mag = np.mean(np.abs(noise))

    # Display in 5 clean columns
    m1, m2, m3, m4, m5 = st.columns(5)

    # Adding logical checks to color the metrics using deltas (green = good, red = bad)
    m1.metric("Embedding Similarity", f"{similarity:.3f}", delta="Blinded",
              delta_color="normal" if similarity < 0.4 else "inverse")
    m2.metric("Visual Fidelity (SSIM)", f"{ssim_score:.3f}", delta="Preserved", delta_color="normal")
    m3.metric("PSNR", f"{psnr_score:.1f} dB")
    m4.metric("Noise Magnitude", f"{noise_mag:.3f}")
    m5.metric("Processing Time", f"{runtime:.2f} s")

    # -----------------------------
    # ADVANCED ANALYSIS (EXPANDER)
    # -----------------------------
    with st.expander("🔍 Advanced: View Perturbation Heatmap"):
        st.markdown(
            "This heatmap isolates the adversarial noise applied by the system. Notice how it strictly targets the facial bounding box to maximize efficiency.")

        heatmap = np.abs(noise)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Center the heatmap nicely
        hm_col1, hm_col2, hm_col3 = st.columns([1, 2, 1])
        with hm_col2:
            st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_container_width=True,
                     caption="Heatmap of L-Infinity Constrained Noise")