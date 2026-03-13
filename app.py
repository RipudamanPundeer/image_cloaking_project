import streamlit as st
import cv2
import numpy as np

from preprocess import load_image
from perturbation import generate_adversarial
from evaluation import get_embedding, cosine_similarity, compute_ssim


st.title("Mobile-Friendly Adversarial Image Cloaking")

st.write("Upload a personal image to generate a protected version.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])


protection_level = st.selectbox(
    "Select Protection Level",
    ["Low","Medium","High"]
)

if protection_level == "Low":
    epsilon = 0.008
elif protection_level == "Medium":
    epsilon = 0.012
else:
    epsilon = 0.018


if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    image = cv2.resize(image,(224,224))

    st.subheader("Original Image")
    st.image(image, channels="BGR")

    if st.button("Generate Protected Image"):

        protected = generate_adversarial(image, epsilon)

        emb1 = get_embedding(image)
        emb2 = get_embedding(protected)

        similarity = cosine_similarity(emb1, emb2).item()

        ssim_score = compute_ssim(image, protected)

        st.subheader("Protected Image")
        st.image(protected, channels="BGR")

        st.write("Embedding Similarity:", similarity)
        st.write("SSIM:", ssim_score)