import os
import cv2
import numpy as np

from preprocess import load_image
from perturbation import generate_adversarial
from evaluation import get_embedding, cosine_similarity


IMAGE_FOLDER = "images"
EPSILON = 0.03


# ---------------------------
# IMAGE TRANSFORMATIONS
# ---------------------------

def resize_transform(image):

    small = cv2.resize(image,(112,112))
    resized = cv2.resize(small,(224,224))

    return resized


def gaussian_blur(image):

    return cv2.GaussianBlur(image,(5,5),0)


def gaussian_noise(image):

    noise = np.random.normal(0,5,image.shape)

    noisy = image + noise

    noisy = np.clip(noisy,0,255).astype(np.uint8)

    return noisy


def brightness_change(image):

    bright = cv2.convertScaleAbs(image,alpha=1.2,beta=20)

    return bright


# ---------------------------
# DATA
# ---------------------------

images = os.listdir(IMAGE_FOLDER)

resize_scores = []
blur_scores = []
noise_scores = []
brightness_scores = []


print("\nRunning robustness tests\n")


for img in images:

    path = os.path.join(IMAGE_FOLDER,img)

    _, image = load_image(path)

    protected = generate_adversarial(image,EPSILON)


    emb_original = get_embedding(image)


    # resize
    resize_img = resize_transform(protected)
    emb_resize = get_embedding(resize_img)

    resize_similarity = cosine_similarity(
        emb_original,
        emb_resize
    ).item()

    resize_scores.append(resize_similarity)


    # blur
    blur_img = gaussian_blur(protected)
    emb_blur = get_embedding(blur_img)

    blur_similarity = cosine_similarity(
        emb_original,
        emb_blur
    ).item()

    blur_scores.append(blur_similarity)


    # noise
    noise_img = gaussian_noise(protected)
    emb_noise = get_embedding(noise_img)

    noise_similarity = cosine_similarity(
        emb_original,
        emb_noise
    ).item()

    noise_scores.append(noise_similarity)


    # brightness
    bright_img = brightness_change(protected)
    emb_bright = get_embedding(bright_img)

    bright_similarity = cosine_similarity(
        emb_original,
        emb_bright
    ).item()

    brightness_scores.append(bright_similarity)


# ---------------------------
# RESULTS
# ---------------------------

print("\nRobustness Results\n")

print("Resize similarity:", np.mean(resize_scores))
print("Blur similarity:", np.mean(blur_scores))
print("Noise similarity:", np.mean(noise_scores))
print("Brightness similarity:", np.mean(brightness_scores))