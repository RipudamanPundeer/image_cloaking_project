import cv2
from preprocess import load_image
from evaluation import get_embedding, cosine_similarity, compute_ssim
from perturbation import generate_adversarial
from robustness import simulate_compression
import numpy as np
# Load image
image = load_image("images/test.jpg")

# Generate cloaked image
epsilon = 0.03   # medium protection 0.08 - 0.018
perturbed = generate_adversarial(image, epsilon)

compressed = simulate_compression(perturbed)

# Compute embeddings
embedding_original = get_embedding(image)
embedding_perturbed = get_embedding(compressed)

# Similarity between embeddings
similarity = cosine_similarity(embedding_original, embedding_perturbed)

# Visual similarity
ssim_score = compute_ssim(image, compressed)

print("Embedding similarity:", similarity.item())
print("SSIM score:", ssim_score)

# Show images
cv2.imshow("Original Image", image)
cv2.imshow("Protected Image", perturbed)

noise = perturbed.astype(float) - image.astype(float)
cv2.imshow("Perturbation Pattern", noise / noise.max())

cv2.imshow("Compressed Protected Image", compressed)

noise = perturbed.astype(float) - image.astype(float)

# normalize for visualization
noise_vis = (noise - noise.min()) / (noise.max() - noise.min())

cv2.imshow("Perturbation Pattern", noise_vis)

cv2.waitKey(0)
cv2.destroyAllWindows()