import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from preprocess import load_image
from perturbation import generate_adversarial
from evaluation import get_embedding, cosine_similarity, compute_ssim


# -----------------------------
# SETTINGS
# -----------------------------

IMAGE_FOLDER = "images"

epsilons = [0.005,0.01,0.015,0.02,0.025,0.03,0.035]


avg_similarity = []
avg_ssim = []
avg_compressed = []
avg_noise = []

all_similarities = []
all_ssim = []


# -----------------------------
# COMPRESSION FUNCTION
# -----------------------------

def compress_image(image, quality=60):

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    _, encimg = cv2.imencode('.jpg', image, encode_param)

    decimg = cv2.imdecode(encimg, 1)

    return decimg


# -----------------------------
# LOAD IMAGE LIST
# -----------------------------

images = sorted(os.listdir(IMAGE_FOLDER))


# -----------------------------
# EXPERIMENT LOOP
# -----------------------------

for epsilon in epsilons:

    sim_scores = []
    ssim_scores = []
    compressed_scores = []
    noise_scores = []

    print("\nTesting epsilon =", epsilon)

    for img in images:

        path = os.path.join(IMAGE_FOLDER, img)

        # skip non-image files
        if not img.lower().endswith((".jpg",".jpeg",".png")):
            continue

        image = load_image(path)

        protected = generate_adversarial(image, epsilon)

        compressed = compress_image(protected)


        # embeddings
        emb_original = get_embedding(image)
        emb_protected = get_embedding(protected)
        emb_compressed = get_embedding(compressed)


        similarity = cosine_similarity(
            emb_original,
            emb_protected
        ).item()

        compression_similarity = cosine_similarity(
            emb_original,
            emb_compressed
        ).item()

        ssim_score = compute_ssim(image, protected)


        # noise magnitude
        noise = protected.astype(np.float32) - image.astype(np.float32)
        noise_mag = np.mean(np.abs(noise))


        sim_scores.append(similarity)
        compressed_scores.append(compression_similarity)
        ssim_scores.append(ssim_score)
        noise_scores.append(noise_mag)

        all_similarities.append(similarity)
        all_ssim.append(ssim_score)


    avg_similarity.append(np.mean(sim_scores))
    avg_ssim.append(np.mean(ssim_scores))
    avg_compressed.append(np.mean(compressed_scores))
    avg_noise.append(np.mean(noise_scores))


print("\nFinished experiments")


# -----------------------------
# GRAPH 1: epsilon vs similarity
# -----------------------------

plt.figure(figsize=(6,4))

plt.plot(epsilons, avg_similarity, marker='o')

plt.title("Embedding Similarity vs Perturbation Strength")

plt.xlabel("Epsilon")
plt.ylabel("Average Similarity")

plt.grid(True)

plt.show()


# -----------------------------
# GRAPH 2: epsilon vs SSIM
# -----------------------------

plt.figure(figsize=(6,4))

plt.plot(epsilons, avg_ssim, marker='o')

plt.title("SSIM vs Perturbation Strength")

plt.xlabel("Epsilon")
plt.ylabel("Average SSIM")

plt.grid(True)

plt.show()


# -----------------------------
# GRAPH 3: compression robustness
# -----------------------------

plt.figure(figsize=(6,4))

plt.plot(epsilons, avg_compressed, marker='o')

plt.title("Similarity After Compression vs Perturbation Strength")

plt.xlabel("Epsilon")
plt.ylabel("Average Similarity")

plt.grid(True)

plt.show()


# -----------------------------
# GRAPH 4: similarity distribution
# -----------------------------

plt.figure(figsize=(6,4))

plt.hist(all_similarities, bins=12)

plt.title("Distribution of Embedding Similarity")

plt.xlabel("Similarity")
plt.ylabel("Number of Images")

plt.grid(True)

plt.show()


# -----------------------------
# GRAPH 5: SSIM vs similarity
# -----------------------------

plt.figure(figsize=(6,4))

plt.scatter(all_ssim, all_similarities)

plt.title("SSIM vs Embedding Similarity")

plt.xlabel("SSIM")
plt.ylabel("Similarity")

plt.grid(True)

plt.show()


# -----------------------------
# GRAPH 6: epsilon vs noise
# -----------------------------

plt.figure(figsize=(6,4))

plt.plot(epsilons, avg_noise, marker='o')

plt.title("Noise Magnitude vs Perturbation Strength")

plt.xlabel("Epsilon")
plt.ylabel("Average Noise Magnitude")

plt.grid(True)

plt.show()