import cv2
import numpy as np

def simulate_compression(image, quality=50):

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)

    decimg = cv2.imdecode(encimg, 1)

    return decimg