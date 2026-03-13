import cv2
from preprocess import load_image
from perturbation import generate_perturbation
from evaluation import get_embedding, cosine_similarity, compute_ssim

# Load image
image = load_image("images/test.jpg")

# Generate cloaked image
perturbed = generate_perturbation(image)

# Compute embeddings
embedding_original = get_embedding(image)
embedding_perturbed = get_embedding(perturbed)

# Similarity between embeddings
similarity = cosine_similarity(embedding_original, embedding_perturbed)

# Visual similarity
ssim_score = compute_ssim(image, perturbed)

print("Embedding similarity:", similarity.item())
print("SSIM score:", ssim_score)

# Show images
cv2.imshow("Original Image", image)
cv2.imshow("Protected Image", perturbed)

cv2.waitKey(0)
cv2.destroyAllWindows()