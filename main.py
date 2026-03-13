import cv2
from preprocess import load_image
from evaluation import get_embedding, cosine_similarity, compute_ssim
from perturbation import generate_adversarial
from robustness import simulate_compression

# Load image
image = load_image("images/test.jpg")

# Generate cloaked image
perturbed = generate_adversarial(image)

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

cv2.waitKey(0)
cv2.destroyAllWindows()