import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from facenet_pytorch import InceptionResnetV1

from preprocess import load_image
from perturbation import generate_adversarial
from evaluation import cosine_similarity


# -----------------------------
# SETTINGS
# -----------------------------

IMAGE_FOLDER = "images"
EPSILON = 0.03

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# MODELS
# -----------------------------

model_attack = InceptionResnetV1(pretrained='vggface2').eval().to(device)

model_test = InceptionResnetV1(pretrained='casia-webface').eval().to(device)


# -----------------------------
# TRANSFORM
# -----------------------------

transform = transforms.Compose([
    transforms.ToTensor()
])


# -----------------------------
# EMBEDDING FUNCTION
# -----------------------------

def get_embedding(model, image):

    image = cv2.resize(image, (160,160))

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(tensor)

    return embedding


# -----------------------------
# DATASET
# -----------------------------

images = sorted(os.listdir(IMAGE_FOLDER))

similarities = []

print("\nTransferability Test\n")


# -----------------------------
# LOOP
# -----------------------------

for img in images:

    if not img.lower().endswith((".jpg",".jpeg",".png")):
        continue

    path = os.path.join(IMAGE_FOLDER, img)

    image = load_image(path)

    protected = generate_adversarial(image, EPSILON)


    emb_original = get_embedding(model_test, image)
    emb_protected = get_embedding(model_test, protected)


    similarity = cosine_similarity(
        emb_original,
        emb_protected
    ).item()

    similarities.append(similarity)

    print(f"{img} -> similarity (transfer test): {similarity:.3f}")


# -----------------------------
# RESULTS
# -----------------------------

print("\nAverage similarity:", np.mean(similarities))