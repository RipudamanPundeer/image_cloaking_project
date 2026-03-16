import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
from skimage.metrics import structural_similarity as ssim


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

    image = cv2.resize(image, (160,160))

    tensor = transform(image).unsqueeze(0).to(device)

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