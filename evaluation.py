import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
from skimage.metrics import structural_similarity as ssim

# Import your face detector
from face_detection import detect_face

# ----------------------------
# MODEL
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ----------------------------
# IMAGE TRANSFORM
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])


# ----------------------------
# EMBEDDING FUNCTION
# ----------------------------
def get_embedding(image):
    # 1. Detect and crop the face JUST like the generator does
    face_box = detect_face(image)

    if face_box is not None:
        x1, y1, x2, y2 = face_box
        # Clamp to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        face = image[y1:y2, x1:x2]

        # Fallback if crop is invalid
        if face.size == 0:
            face = image
    else:
        # Fallback if no face detected
        face = image

        # 2. Resize ONLY the face to 160x160 for FaceNet
    face = cv2.resize(face, (160, 160))

    tensor = transform(face).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(tensor)

    return embedding


# ----------------------------
# COSINE SIMILARITY
# ----------------------------
def cosine_similarity(e1, e2):
    sim = torch.nn.functional.cosine_similarity(e1, e2)
    return sim


# ----------------------------
# SSIM
# ----------------------------
def compute_ssim(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    score = ssim(
        img1,
        img2,
        channel_axis=2,
        data_range=255
    )

    return score