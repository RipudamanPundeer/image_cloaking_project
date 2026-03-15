import os
import numpy as np
import matplotlib.pyplot as plt

from preprocess import load_image
from perturbation import generate_adversarial
from evaluation import get_embedding, cosine_similarity, compute_ssim


# ---------------------------
# SETTINGS
# ---------------------------

IMAGE_FOLDER = "images"

# perturbation strengths to test
epsilons = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]


# ---------------------------
# DATA COLLECTION
# ---------------------------

images = os.listdir(IMAGE_FOLDER)

similarity_results = []
ssim_results = []

all_similarities = []


# ---------------------------
# EXPERIMENT LOOP
# ---------------------------

for epsilon in epsilons:

    print("\nTesting epsilon =", epsilon)

    sim_scores = []
    ssim_scores = []

    for img in images:

        path = os.path.join(IMAGE_FOLDER, img)

        # load image (resized version for model)
        original, image = load_image(path)

        # generate protected image
        protected = generate_adversarial(image, epsilon)

        # compute embeddings
        emb1 = get_embedding(image)
        emb2 = get_embedding(protected)

        sim = cosine_similarity(emb1, emb2).item()
        ssim_score = compute_ssim(image, protected)

        print(f"{img} -> similarity: {sim:.4f}  SSIM: {ssim_score:.4f}")

        sim_scores.append(sim)
        ssim_scores.append(ssim_score)

        all_similarities.append(sim)

    # compute average metrics for this epsilon
    similarity_results.append(np.mean(sim_scores))
    ssim_results.append(np.mean(ssim_scores))


# ---------------------------
# SUMMARY STATISTICS
# ---------------------------

print("\n==============================")
print("EXPERIMENT SUMMARY")
print("==============================")

print("Average similarity:", np.mean(all_similarities))
print("Min similarity:", np.min(all_similarities))
print("Max similarity:", np.max(all_similarities))


# ---------------------------
# GRAPH 1
# Similarity vs epsilon
# ---------------------------

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epsilons, similarity_results, marker='o')
plt.title("Embedding Similarity vs Perturbation Strength")
plt.xlabel("Epsilon")
plt.ylabel("Average Similarity")
plt.grid(True)


# ---------------------------
# GRAPH 2
# SSIM vs epsilon
# ---------------------------

plt.subplot(1,2,2)
plt.plot(epsilons, ssim_results, marker='o')
plt.title("SSIM vs Perturbation Strength")
plt.xlabel("Epsilon")
plt.ylabel("Average SSIM")
plt.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------
# HISTOGRAM OF SIMILARITIES
# ---------------------------

plt.figure(figsize=(6,5))

plt.hist(all_similarities, bins=10)

plt.title("Distribution of Embedding Similarities")
plt.xlabel("Similarity")
plt.ylabel("Number of Images")

plt.grid(True)

plt.show()