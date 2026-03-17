import cv2
import numpy as np

from preprocess import load_image
from perturbation import generate_adversarial
from evaluation import get_embedding, cosine_similarity, compute_ssim
from face_detection import detect_face


# ----------------------------
# SETTINGS
# ----------------------------

IMAGE_PATH = "images/IMG_3055.JPG"
EPSILON = 0.03


# ----------------------------
# LOAD IMAGE
# ----------------------------

original = load_image(IMAGE_PATH)

display_original = original.copy()


# ----------------------------
# FACE DETECTION
# ----------------------------

face_box = detect_face(original)

if face_box is not None:

    x1, y1, x2, y2 = face_box

    print("Face box:", x1, y1, x2, y2)

    h, w = original.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    cv2.rectangle(display_original, (x1,y1), (x2,y2), (0,255,0), 3)


# ----------------------------
# GENERATE ADVERSARIAL IMAGE
# ----------------------------

protected = generate_adversarial(original, EPSILON)


# ----------------------------
# COMPUTE NOISE
# ----------------------------

noise = protected.astype(float) - original.astype(float)

noise_magnitude = np.mean(np.abs(noise))

print("Noise magnitude:", noise_magnitude)


# ----------------------------
# EVALUATION METRICS
# ----------------------------

embedding_original = get_embedding(original)
embedding_protected = get_embedding(protected)

similarity = cosine_similarity(
    embedding_original,
    embedding_protected
).item()

ssim_score = compute_ssim(original, protected)


print("Embedding similarity:", similarity)
print("SSIM score:", ssim_score)


# ----------------------------
# HEATMAP
# ----------------------------

noise_abs = np.abs(noise)

heatmap = noise_abs / (noise_abs.max() + 1e-8)

heatmap = (heatmap * 255).astype(np.uint8)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


# ----------------------------
# COMPARISON IMAGE
# ----------------------------

comparison = np.hstack((original, protected))


# ----------------------------
# DISPLAY WINDOWS
# ----------------------------

cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Protected Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Perturbation Heatmap", cv2.WINDOW_NORMAL)
cv2.namedWindow("Original vs Protected", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Original Image", 700, 700)
cv2.resizeWindow("Protected Image", 700, 700)
cv2.resizeWindow("Perturbation Heatmap", 700, 700)
cv2.resizeWindow("Original vs Protected", 1200, 600)

cv2.imshow("Original Image", display_original)
cv2.imshow("Protected Image", protected)
cv2.imshow("Perturbation Heatmap", heatmap)
cv2.imshow("Original vs Protected", comparison)

cv2.waitKey(0)
cv2.destroyAllWindows()