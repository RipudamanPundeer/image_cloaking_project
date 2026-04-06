import streamlit as st
import requests
import cv2
import numpy as np
import io
from PIL import Image

# --- 1. UI CONFIG ---
st.set_page_config(page_title="Privacy Cloak", page_icon="🛡️", layout="centered")

# CSS to center the mobile-style container and format the slider
st.markdown("""
    <style>
    .main { max-width: 450px; margin: 0 auto; }
    .stButton>button { width: 100%; border-radius: 12px; height: 3.5em; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Privacy Cloak")
st.caption("Edge-Assisted Identity Protection")

# --- 2. NUMERICAL CONTROLS ---
# Exposing the actual Epsilon value to the user/panel
epsilon_val = st.slider(
    "Protection Strength (Epsilon)",
    min_value=0.01,
    max_value=0.10,
    value=0.05,
    step=0.01,
    help="Higher values increase protection but introduce more visible noise."
)

st.divider()

# --- 3. PHOTO SELECTION ---
uploaded_file = st.file_uploader("Select Photo", type=["jpg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    img_bytes = uploaded_file.getvalue()

    # Constrain the image size so it doesn't fill the screen
    _, mid_col, _ = st.columns([1, 4, 1])
    with mid_col:
        st.image(Image.open(io.BytesIO(img_bytes)), caption="Source Photo", width=300)

    # --- 4. THE ACTION ---
    if st.button("Generate Privacy Shield 🔒"):
        with st.spinner("Offloading to Edge Node..."):
            try:
                files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
                # Pass the exact slider value to the API
                data = {"epsilon": epsilon_val, "iters": 10}

                # Update this IP to your laptop's IP if testing on a real phone on the same WiFi
                response = requests.post("http://127.0.0.1:8000/protect", files=files, data=data)

                if response.status_code == 200:
                    st.success("Identity Successfully Cloaked!")

                    # Decode result
                    prot_arr = np.frombuffer(response.content, np.uint8)
                    prot_img = cv2.imdecode(prot_arr, cv2.IMREAD_COLOR)

                    # Center the result
                    _, res_col, _ = st.columns([1, 4, 1])
                    with res_col:
                        st.image(cv2.cvtColor(prot_img, cv2.COLOR_BGR2RGB), caption=f"Protected (ε={epsilon_val})",
                                 width=300)

                    # Download button
                    st.download_button(
                        label="Save to Gallery 📥",
                        data=response.content,
                        file_name=f"protected_eps_{epsilon_val}.png",
                        mime="image/png"
                    )
                else:
                    st.error("API Error. Check backend logs.")
            except Exception as e:
                st.error(f"Connection Failed: Ensure mobile_api.py is running.")

st.caption("System Status: Distributed Edge Active")