import os
import cv2
import csv
import shutil
import numpy as np
import tensorflow as tf

# PyTorch Imports
import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1

# DeepFace Imports
from deepface import DeepFace

# Your custom modules
from preprocess import load_image
from perturbation import generate_adversarial
from face_detection import detect_face
from evaluation import compute_psnr

# --- STOP TENSORFLOW FROM HOGGING ALL CPU ---
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# -----------------------------
# SETTINGS
# -----------------------------
IMAGE_FOLDER = "images"
TEMP_FOLDER = "temp_crops"
EPSILON = 0.05
RESULTS_FILE = "master_transferability_results.csv"

DEEPFACE_MODELS = ["Facenet", "ArcFace", "VGG-Face"]
ALL_METRICS = ["Facenet (White-Box)", "Facenet-CASIA (Cross-Dataset)", "ArcFace (Cross-Arch)", "VGG-Face (Cross-Arch)"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_sim_np(emb1, emb2):
    a = np.array(emb1)
    b = np.array(emb2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def run_master_transfer_test():
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not images:
        print("No images found.")
        return

    print(f"\nStarting Master Transferability Test on {len(images)} images...")
    print("Strategy: Disk-Caching, Cross-Dataset (PyTorch), and Cross-Architecture (TensorFlow)")
    print("-" * 85)

    if os.path.exists(TEMP_FOLDER):
        shutil.rmtree(TEMP_FOLDER)
    os.makedirs(TEMP_FOLDER)

    results = {img: {} for img in images}
    sim_trackers = {metric: [] for metric in ALL_METRICS}
    valid_images = []

    # ==========================================
    # STEP 1: GENERATE & CROP (Standard PyTorch PGD)
    # ==========================================
    print("Step 1/3: Generating Fast PGD Attacks and Saving Perfect Crops...")

    # surrogate_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    # transform = transforms.Compose([transforms.ToTensor()])

    for i, img_name in enumerate(images, 1):
        path = os.path.join(IMAGE_FOLDER, img_name)
        try:
            # 1. Load Original
            image = load_image(path)

            # 2. Get Face Bounding Box (needed for cropping the result later)
            face_box = detect_face(image)
            if face_box is None:
                continue

            x1, y1, x2, y2 = face_box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            # 3. GENERATE ATTACK (Using your new wrapper)
            # This handles detection/resizing/PGD/stitching internally
            protected_full_image = generate_adversarial(image, epsilon=EPSILON, steps=10)

            # 4. PREPARE CROPS FOR EVALUATION
            # We need 160x160 crops for the transferability metrics in Step 2 & 3
            orig_face = image[y1:y2, x1:x2]
            prot_face = protected_full_image[y1:y2, x1:x2]

            orig_crop_160 = cv2.resize(orig_face, (160, 160))
            prot_crop_160 = cv2.resize(prot_face, (160, 160))

            # 5. Save the isolated 160x160 crops to Disk for DeepFace
            # Convert RGB to BGR before saving to hard drive
            cv2.imwrite(os.path.join(TEMP_FOLDER, f"orig_{img_name}"), cv2.cvtColor(orig_crop_160, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(TEMP_FOLDER, f"prot_{img_name}"), cv2.cvtColor(prot_crop_160, cv2.COLOR_RGB2BGR))

            valid_images.append(img_name)
            print(f"  [{i}/{len(images)}] Generated & Cached crops for {img_name}")

        except Exception as e:
            print(f"  [{i}/{len(images)}] Error generating {img_name}: {e}")

    # Free up memory before TensorFlow takes over
    # del surrogate_mode
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # ==========================================
    # STEP 2: CROSS-DATASET (PyTorch Engine)
    # ==========================================
    print("\nStep 2/3: Evaluating Cross-Dataset Transferability (CASIA-WebFace)...")

    # Load CASIA model into memory
    casia_model = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
    transform = transforms.Compose([transforms.ToTensor()])

    for i, img_name in enumerate(valid_images, 1):
        orig_path = os.path.join(TEMP_FOLDER, f"orig_{img_name}")
        prot_path = os.path.join(TEMP_FOLDER, f"prot_{img_name}")

        # Load and convert BGR (OpenCV) to RGB (PyTorch standard)
        orig_img = cv2.cvtColor(cv2.imread(orig_path), cv2.COLOR_BGR2RGB)
        prot_img = cv2.cvtColor(cv2.imread(prot_path), cv2.COLOR_BGR2RGB)

        # Resize to 160x160 as required by InceptionResnetV1
        orig_resized = cv2.resize(orig_img, (160, 160))
        prot_resized = cv2.resize(prot_img, (160, 160))

        t_orig = transform(orig_resized).unsqueeze(0).to(device)
        t_prot = transform(prot_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            emb_orig = casia_model(t_orig)
            emb_prot = casia_model(t_prot)

        sim = torch.nn.functional.cosine_similarity(emb_orig, emb_prot).item()

        results[img_name]["Facenet-CASIA (Cross-Dataset)"] = round(sim, 4)
        sim_trackers["Facenet-CASIA (Cross-Dataset)"].append(sim)
        print(f"  [{i}/{len(valid_images)}] {img_name} -> CASIA Similarity: {sim:.3f}")

    # --- MEMORY WIPE ---
    # Delete the PyTorch model to free up RAM before TensorFlow takes over
    del casia_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==========================================
    # STEP 3: CROSS-ARCHITECTURE (TensorFlow)
    # ==========================================
    print("\nStep 3/3: Evaluating Cross-Architecture (DeepFace: Facenet, ArcFace, VGG-Face)...")

    for model_name in DEEPFACE_MODELS:
        print(f"\n--- Loading {model_name} into memory ---")

        # Map deepface model names to our formatted CSV column names
        metric_name = "Facenet (White-Box)" if model_name == "Facenet" else f"{model_name} (Cross-Arch)"

        for i, img_name in enumerate(valid_images, 1):
            orig_path = os.path.join(TEMP_FOLDER, f"orig_{img_name}")
            prot_path = os.path.join(TEMP_FOLDER, f"prot_{img_name}")

            try:
                # detector_backend="skip" relies on our perfect crops
                orig_obj = DeepFace.represent(img_path=orig_path, model_name=model_name, detector_backend="skip",
                                              enforce_detection=False)
                prot_obj = DeepFace.represent(img_path=prot_path, model_name=model_name, detector_backend="skip",
                                              enforce_detection=False)

                sim = cosine_sim_np(orig_obj[0]["embedding"], prot_obj[0]["embedding"])

                results[img_name][metric_name] = round(sim, 4)
                sim_trackers[metric_name].append(sim)

                print(f"  [{i}/{len(valid_images)}] {img_name} -> Similarity: {sim:.3f}")

            except Exception as e:
                print(f"  [{i}/{len(valid_images)}] DeepFace Error on {img_name}: {e}")

    # ==========================================
    # CLEANUP & SAVE
    # ==========================================
    if os.path.exists(TEMP_FOLDER):
        shutil.rmtree(TEMP_FOLDER)

    csv_rows = []
    for img_name in valid_images:
        row = {"Image": img_name}
        for metric in ALL_METRICS:
            row[metric] = results[img_name].get(metric, "ERROR")
        csv_rows.append(row)

    if csv_rows:
        with open(RESULTS_FILE, mode='w', newline='') as file:
            fieldnames = ["Image"] + ALL_METRICS
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    print("\n" + "=" * 75)
    print("FINAL MASTER TRANSFERABILITY SCORES")
    print("=" * 75)
    for metric in ALL_METRICS:
        if sim_trackers[metric]:
            print(f"{metric.ljust(35)} : {np.mean(sim_trackers[metric]): .4f}")
    print("=" * 75)


if __name__ == "__main__":
    run_master_transfer_test()