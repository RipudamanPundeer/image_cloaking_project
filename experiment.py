import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from preprocess import load_image
from perturbation import generate_adversarial
from evaluation import get_embedding, cosine_similarity, compute_ssim


image_folder = "images"

epsilons = [0.005,0.006,0.008,0.010,0.012,0.015,0.018,0.02,0.025]

similarity_results = []
ssim_results = []


images = os.listdir(image_folder)

for epsilon in epsilons:

    sim_scores = []
    ssim_scores = []

    for img in images:

        path = os.path.join(image_folder,img)

        _,image = load_image(path)

        protected = generate_adversarial(image,epsilon)

        emb1 = get_embedding(image)
        emb2 = get_embedding(protected)

        sim = cosine_similarity(emb1,emb2).item()
        ssim_score = compute_ssim(image,protected)

        sim_scores.append(sim)
        ssim_scores.append(ssim_score)

    similarity_results.append(np.mean(sim_scores))
    ssim_results.append(np.mean(ssim_scores))


print("Similarity:",similarity_results)
print("SSIM:",ssim_results)

plt.figure()

plt.subplot(1,2,1)
plt.plot(epsilons,similarity_results,marker='o')
plt.title("Embedding Similarity vs Perturbation Strength")
plt.xlabel("Epsilon")
plt.ylabel("Similarity")

plt.subplot(1,2,2)
plt.plot(epsilons,ssim_results,marker='o')
plt.title("SSIM vs Perturbation Strength")
plt.xlabel("Epsilon")
plt.ylabel("SSIM")

plt.show()