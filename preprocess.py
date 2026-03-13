import cv2

def load_image(path):

    original = cv2.imread(path)

    resized = cv2.resize(original,(224,224))

    return original, resized