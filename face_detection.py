from facenet_pytorch import MTCNN
import torch
import cv2


# ----------------------------
# DEVICE
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# FACE DETECTOR
# ----------------------------

mtcnn = MTCNN(
    keep_all=False,
    device=device
)


# ----------------------------
# FACE DETECTION FUNCTION
# ----------------------------

def detect_face(image):

    boxes, probs = mtcnn.detect(image)

    if boxes is None:
        return None

    # use first detected face
    x1, y1, x2, y2 = boxes[0]

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    h, w = image.shape[:2]

    # clamp to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return x1, y1, x2, y2