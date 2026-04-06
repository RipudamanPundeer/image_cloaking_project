import cv2


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not load image: {path}")

    # CRITICAL FIX: Always return RGB so FaceNet and MTCNN read correct colors globally
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)