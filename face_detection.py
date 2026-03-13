from facenet_pytorch import MTCNN
import cv2

mtcnn = MTCNN(keep_all=False)

def detect_face(image):
    boxes, _ = mtcnn.detect(image)

    if boxes is None:
        return None

    x1, y1, x2, y2 = boxes[0].astype(int)
    return x1, y1, x2, y2