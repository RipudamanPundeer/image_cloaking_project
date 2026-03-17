import os
import time
import cv2
import csv
import numpy as np

from preprocess import load_image
from perturbation import generate_adversarial
from evaluation import get_embedding, cosine_similarity

# -----------------------------
# SETTINGS
# -----------------------------
IMAGE_FOLDER = "images"
EPSILON = 0.03
RESULTS_FILE = "robustness_results.csv"


# ---------------------------
# IMAGE TRANSFORMATIONS
# ---------------------------
def resize_transform(image):
    # Simulates social media downscaling and upscaling
    small = cv2.resize(image, (112, 112))
    return cv2.resize(small, (224, 224))


def gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def gaussian_noise(image):
    noise = np.random.normal(0, 5, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def brightness_change(image):
    return cv2.convertScaleAbs(image, alpha=1.2, beta=20)


def simulate_compression(image, quality=50):
    # Standard JPEG compression simulation
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, 1)


# ---------------------------
# MAIN EXECUTION
# ---------------------------
def run_robustness_tests():
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not images:
        print(f"No images found in {IMAGE_FOLDER}")
        return

    print(f"\nStarting Robustness Tests on {len(images)} images (Epsilon: {EPSILON})")
    print("-" * 70)

    # Data tracking
    metrics = {
        "Baseline": [],
        "Compression": [],
        "Resize": [],
        "Blur": [],
        "Noise": [],
        "Brightness": []
    }

    csv_rows = []
    start_time = time.time()

    for i, img_name in enumerate(images, 1):
        path = os.path.join(IMAGE_FOLDER, img_name)

        try:
            # 1. Load and Protect
            image = load_image(path)
            protected = generate_adversarial(image, EPSILON)

            # Original Embedding (Anchor)
            emb_original = get_embedding(image)

            # 2. Apply Transformations to the PROTECTED image
            transforms = {
                "Baseline": protected,
                "Compression": simulate_compression(protected, quality=60),
                "Resize": resize_transform(protected),
                "Blur": gaussian_blur(protected),
                "Noise": gaussian_noise(protected),
                "Brightness": brightness_change(protected)
            }

            # 3. Evaluate each transformation
            row_data = {"Image": img_name}
            log_str = f"[{i}/{len(images)}] {img_name} | "

            for t_name, t_img in transforms.items():
                emb_transformed = get_embedding(t_img)
                sim = cosine_similarity(emb_original, emb_transformed).item()

                metrics[t_name].append(sim)
                row_data[t_name] = round(sim, 4)

                # Add to log string (abbreviated for terminal)
                log_str += f"{t_name[:4]}: {sim: .2f} | "

            csv_rows.append(row_data)
            print(log_str[:-3])  # Print progress

        except Exception as e:
            print(f"[{i}/{len(images)}] Error processing {img_name}: {e}")

    # ---------------------------
    # SAVE CSV AND PRINT RESULTS
    # ---------------------------
    if csv_rows:
        with open(RESULTS_FILE, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["Image", "Baseline", "Compression", "Resize", "Blur", "Noise",
                                                      "Brightness"])
            writer.writeheader()
            writer.writerows(csv_rows)

    total_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("ROBUSTNESS TEST RESULTS (Average Similarity)")
    print("*(Lower is better. Closer to 1.0 means attack failed)*")
    print("=" * 50)

    for t_name, scores in metrics.items():
        if scores:
            avg_sim = np.mean(scores)
            print(f"{t_name.ljust(15)} : {avg_sim: .4f}")

    print("-" * 50)
    print(f"Results saved to    : {RESULTS_FILE}")
    print(f"Total Time          : {total_time:.2f} seconds")
    print("=" * 50)


if __name__ == "__main__":
    run_robustness_tests()