import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from preprocess import load_image
from perturbation import generate_adversarial
from evaluation import get_embedding, cosine_similarity, compute_ssim


IMAGE_FOLDER = "images"

epsilons = [0.005,0.01,0.015,0.02,0.025,0.03]

avg_similarity = []
avg_ssim = []
avg_compressed = []


def compress_image(image, quality=60):

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    _, encimg = cv2.imencode('.jpg', image, encode_param)

    decimg = cv2.imdecode(encimg, 1)

    return decimg


images = os.listdir(IMAGE_FOLDER)


for epsilon in epsilons:

    sim_scores = []
    ssim_scores = []
    compressed_scores = []

    print("\nTesting epsilon =", epsilon)

    for img in images:

        path = os.path.join(IMAGE_FOLDER,img)

        original, image = load_image(path)

        protected = generate_adversarial(image,epsilon)

        compressed = compress_image(protected)


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

        ssim_score = compute_ssim(image,protected)


        sim_scores.append(similarity)
        compressed_scores.append(compression_similarity)
        ssim_scores.append(ssim_score)


    avg_similarity.append(np.mean(sim_scores))
    avg_compressed.append(np.mean(compressed_scores))
    avg_ssim.append(np.mean(ssim_scores))


print("\nFinished experiments")



# -----------------------------
# GRAPH 1: epsilon vs similarity
# -----------------------------

plt.figure(figsize=(6,4))

plt.plot(epsilons,avg_similarity,marker='o')

plt.title("Embedding Similarity vs Perturbation Strength")

plt.xlabel("Epsilon")
plt.ylabel("Average Similarity")

plt.grid(True)

plt.show()



# -----------------------------
# GRAPH 2: epsilon vs SSIM
# -----------------------------

plt.figure(figsize=(6,4))

plt.plot(epsilons,avg_ssim,marker='o')

plt.title("SSIM vs Perturbation Strength")

plt.xlabel("Epsilon")
plt.ylabel("Average SSIM")

plt.grid(True)

plt.show()



# -----------------------------
# GRAPH 3: epsilon vs compression similarity
# -----------------------------

plt.figure(figsize=(6,4))

plt.plot(epsilons,avg_compressed,marker='o')

plt.title("Similarity After Compression vs Perturbation Strength")

plt.xlabel("Epsilon")
plt.ylabel("Average Similarity")

plt.grid(True)

plt.show()