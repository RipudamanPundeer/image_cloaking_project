import cv2
import numpy as np

from preprocess import load_image
from perturbation import generate_adversarial
from evaluation import get_embedding, cosine_similarity, compute_ssim


# ----------------------------
# SETTINGS
# ----------------------------

IMAGE_PATH = "images/IMG_6032.JPG"
EPSILON = 0.012


# ----------------------------
# LOAD IMAGE
# ----------------------------

original, resized = load_image(IMAGE_PATH)


# ----------------------------
# GENERATE ADVERSARIAL IMAGE
# ----------------------------

protected_resized = generate_adversarial(resized, EPSILON)


# ----------------------------
# COMPUTE PERTURBATION
# ----------------------------

noise = protected_resized.astype(float) - resized.astype(float)


# ----------------------------
# UPSCALE NOISE TO ORIGINAL SIZE
# ----------------------------

noise_upscaled = cv2.resize(
    noise,
    (original.shape[1], original.shape[0]),
    interpolation=cv2.INTER_LINEAR
)


# ----------------------------
# APPLY PERTURBATION
# ----------------------------

protected_full = original.astype(float) + noise_upscaled
protected_full = np.clip(protected_full, 0, 255).astype(np.uint8)


# ----------------------------
# EVALUATION METRICS
# ----------------------------

embedding_original = get_embedding(resized)
embedding_protected = get_embedding(protected_resized)

similarity = cosine_similarity(
    embedding_original,
    embedding_protected
).item()

ssim_score = compute_ssim(resized, protected_resized)


print("Embedding similarity:", similarity)
print("SSIM score:", ssim_score)


# ----------------------------
# HEATMAP VISUALIZATION
# ----------------------------

noise_abs = np.abs(noise)

max_val = noise_abs.max()

if max_val == 0:
    noise_vis = noise_abs
else:
    noise_vis = noise_abs / max_val


heatmap = (noise_vis * 255).astype(np.uint8)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


# ----------------------------
# DISPLAY WINDOWS
# ----------------------------

cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Protected Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Perturbation Heatmap", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Original Image", 700, 700)
cv2.resizeWindow("Protected Image", 700, 700)
cv2.resizeWindow("Perturbation Heatmap", 700, 700)

cv2.imshow("Original Image", original)
cv2.imshow("Protected Image", protected_full)
cv2.imshow("Perturbation Heatmap", heatmap)


cv2.waitKey(0)
cv2.destroyAllWindows()