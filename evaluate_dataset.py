import os
import time
import numpy as np
from face_detection import detect_face
from preprocess import load_image
from perturbation import generate_adversarial
from evaluation import get_embedding, cosine_similarity, compute_ssim, compute_psnr
# -----------------------------
# CONFIGURATION
# -----------------------------
IMAGE_FOLDER = "images"  # Point this to your folder of 124 photos
EPSILON = 0.025  # Set your target perturbation strength


def evaluate_all():
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not images:
        print(f"No images found in {IMAGE_FOLDER}")
        return

    print(f"Found {len(images)} images. Starting batch evaluation (Epsilon: {EPSILON})...\n")

    similarities = []
    ssim_scores = []
    psnr_scores = []

    start_time = time.time()

    for i, img_name in enumerate(images, 1):
        path = os.path.join(IMAGE_FOLDER, img_name)

        try:
            # 1. Load and process
            image = load_image(path)
            # skip logic!
            if detect_face(image) is None:
                print(f"  -> Skipping {img_name}: No face detected.")
                continue
            protected = generate_adversarial(image, EPSILON)

            # 2. Get embeddings
            emb_original = get_embedding(image)
            emb_protected = get_embedding(protected)

            # 3. Calculate metrics
            sim = cosine_similarity(emb_original, emb_protected).item()
            ssim_val = compute_ssim(image, protected)
            psnr_val = compute_psnr(image, protected)

            similarities.append(sim)
            ssim_scores.append(ssim_val)
            if psnr_val!=float('inf'):
                psnr_scores.append(psnr_val)
            else:
                psnr_scores.append(50)

            # Print progress
            print(f"[{i}/{len(images)}] {img_name} | Sim: {sim:.3f} | SSIM: {ssim_val:.3f} | PSNR: {psnr_val:.2f}dB")
        except Exception as e:
            print(f"[{i}/{len(images)}] Error processing {img_name}: {e}")

    # -----------------------------
    # FINAL AGGREGATED RESULTS
    # -----------------------------
    total_time = time.time() - start_time

    print("\n" + "=" * 40)
    print("FINAL AGGREGATED RESULTS")
    print("=" * 40)
    print(f"Total Images Processed : {len(similarities)}")
    print(f"Average Similarity     : {np.mean(similarities):.4f} (Lower is better)")
    print(f"Average SSIM           : {np.mean(ssim_scores):.4f} (Higher is better)")
    print(f"Average PSNR           : {np.mean(psnr_scores):.2f} dB (Above 30dB is excellent)")
    print(f"Total Processing Time  : {total_time:.2f} seconds")
    print("=" * 40)


if __name__ == "__main__":
    evaluate_all()