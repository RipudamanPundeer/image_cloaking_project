import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from preprocess import load_image
from perturbation import generate_adversarial
from evaluation import get_embedding, cosine_similarity, compute_ssim, compute_psnr

# -----------------------------
# SETTINGS
# -----------------------------
IMAGE_FOLDER = "images"
epsilons = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035]

# Set Matplotlib style for beautiful, colorful graphs
plt.style.use('ggplot')


# -----------------------------
# COMPRESSION FUNCTION
# -----------------------------
def simulate_compression(image, quality=60):  # Note: robustness_test uses quality=50
    # 1. Convert RGB to BGR for OpenCV encoder
    bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 2. Compress
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', bgr_img, encode_param)
    decimg = cv2.imdecode(encimg, 1)

    # 3. Convert back to RGB for the PyTorch metrics
    return cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)


# -----------------------------
# MAIN EXPERIMENT RUNNER
# -----------------------------
def run_experiments():
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not images:
        print(f"No images found in {IMAGE_FOLDER}")
        return

    print(f"\nStarting Epsilon Scaling Experiments on {len(images)} images...")
    print("*(Undetected faces will be automatically filtered from the averages)*\n")

    # Arrays to hold average metrics for each epsilon
    avg_similarity = []
    avg_ssim = []
    avg_psnr = []
    avg_compressed = []
    avg_noise = []

    # For scatter plotting at the end
    scatter_ssim = []
    scatter_sim = []
    scatter_eps = []

    total_start_time = time.time()

    for eps in epsilons:
        print(f"--- Testing Epsilon: {eps:.3f} ---")

        sim_scores = []
        ssim_scores = []
        psnr_scores = []
        comp_scores = []
        noise_scores = []

        for i, img_name in enumerate(images, 1):
            path = os.path.join(IMAGE_FOLDER, img_name)

            try:
                # 1. Load and Protect
                image = load_image(path)
                protected = generate_adversarial(image, epsilon=eps)

                # 2. Check if attack actually ran (Filter out undetected faces)
                psnr_score = compute_psnr(image, protected)
                if psnr_score == float('inf'):
                    continue  # Skip to next image

                # 3. Compress the protected image
                compressed = simulate_compression(protected, quality=60)

                # 4. Extract embeddings
                emb_original = get_embedding(image)
                emb_protected = get_embedding(protected)
                emb_compressed = get_embedding(compressed)

                # 5. Calculate Metrics
                sim = cosine_similarity(emb_original, emb_protected).item()
                comp_sim = cosine_similarity(emb_original, emb_compressed).item()
                ssim_score = compute_ssim(image, protected)

                noise = protected.astype(np.float32) - image.astype(np.float32)
                noise_mag = np.mean(np.abs(noise))

                # Store valid metrics
                sim_scores.append(sim)
                ssim_scores.append(ssim_score)
                psnr_scores.append(psnr_score)
                comp_scores.append(comp_sim)
                noise_scores.append(noise_mag)

                # Store data for the scatter plot
                scatter_ssim.append(ssim_score)
                scatter_sim.append(sim)
                scatter_eps.append(eps)

            except Exception as e:
                print(f"Error on {img_name}: {e}")

        # Calculate averages for this epsilon step
        if sim_scores:
            avg_similarity.append(np.mean(sim_scores))
            avg_ssim.append(np.mean(ssim_scores))
            avg_psnr.append(np.mean(psnr_scores))
            avg_compressed.append(np.mean(comp_scores))
            avg_noise.append(np.mean(noise_scores))

            print(
                f"   Avg Sim: {avg_similarity[-1]:.3f} | Avg SSIM: {avg_ssim[-1]:.3f} | Avg PSNR: {avg_psnr[-1]:.1f}dB\n")
        else:
            print("   No valid images processed for this epsilon.\n")

    print(f"Experiments completed in {time.time() - total_start_time:.2f} seconds.")
    print("Generating Graphs...\n")

    # -----------------------------
    # GRAPH GENERATION
    # -----------------------------

    # 1: Epsilon vs Similarity (The Core Attack Success)
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, avg_similarity, marker='o', color='#E24A33', linewidth=2, markersize=8)
    plt.title("Attack Effectiveness: Similarity vs. Perturbation Strength", fontsize=14, fontweight='bold')
    plt.xlabel("Perturbation Strength (Epsilon)", fontsize=12)
    plt.ylabel("Cosine Similarity (Lower is Better)", fontsize=12)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)  # Zero line reference
    plt.tight_layout()
    plt.show()

    # 2: Epsilon vs Visual Quality (SSIM & PSNR)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = '#348ABD'
    ax1.set_xlabel('Perturbation Strength (Epsilon)', fontsize=12)
    ax1.set_ylabel('SSIM', color=color, fontsize=12)
    ax1.plot(epsilons, avg_ssim, marker='s', color=color, linewidth=2, markersize=8, label="SSIM")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = '#8EBA42'
    ax2.set_ylabel('PSNR (dB)', color=color, fontsize=12)
    ax2.plot(epsilons, avg_psnr, marker='^', color=color, linewidth=2, markersize=8, label="PSNR")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Visual Quality Preservation vs. Perturbation Strength", fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.show()

    # 3: Epsilon vs Compression Robustness
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, avg_similarity, marker='o', color='#E24A33', linewidth=2, label="Original Protected")
    plt.plot(epsilons, avg_compressed, marker='D', color='#988ED5', linewidth=2, linestyle='--',
             label="After JPEG Compression")
    plt.title("Robustness: Attack Survival After Compression", fontsize=14, fontweight='bold')
    plt.xlabel("Perturbation Strength (Epsilon)", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # 4: Epsilon vs Noise Magnitude
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, avg_noise, marker='p', color='#FBC15E', linewidth=2, markersize=8)
    plt.title("System Behavior: Noise Magnitude vs. Epsilon", fontsize=14, fontweight='bold')
    plt.xlabel("Perturbation Strength (Epsilon)", fontsize=12)
    plt.ylabel("Average Pixel Noise Magnitude", fontsize=12)
    plt.tight_layout()
    plt.show()

    # 5: The Ultimate Trade-off (Scatter Plot)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(scatter_ssim, scatter_sim, c=scatter_eps, cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Perturbation Strength (Epsilon)')
    plt.title("The Trade-off: Visual Quality (SSIM) vs. Attack Success", fontsize=14, fontweight='bold')
    plt.xlabel("Visual Quality (SSIM - Closer to 1.0 is better)", fontsize=12)
    plt.ylabel("Embedding Similarity (Lower is better)", fontsize=12)
    plt.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='Typical Recognition Threshold')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiments()