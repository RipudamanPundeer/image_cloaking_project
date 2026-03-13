import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
import numpy as np
from face_detection import detect_face
import cv2

model = InceptionResnetV1(pretrained='vggface2').eval()

transform = transforms.Compose([
    transforms.ToTensor()
])



def generate_adversarial(image, epsilon=0.012):

    face_box = detect_face(image)

    if face_box is None:
        return image

    x1, y1, x2, y2 = face_box

    face = image[y1:y2, x1:x2]

    # resize face to model size
    face = cv2.resize(face, (160,160))

    tensor = transform(face).unsqueeze(0)
    tensor.requires_grad = True

    original_embedding = model(tensor).detach()
    embedding = model(tensor)

    loss = -torch.nn.functional.cosine_similarity(
        embedding, original_embedding
    ).mean()

    loss.backward()

    perturbation = epsilon * tensor.grad.sign()

    adv_face = tensor + perturbation
    adv_face = torch.clamp(adv_face, 0, 1)

    adv_face = adv_face.squeeze().permute(1,2,0).detach().numpy()
    adv_face = (adv_face * 255).astype(np.uint8)

    # resize back to original face region size
    adv_face = cv2.resize(adv_face, (x2-x1, y2-y1))

    adv_image = image.copy()
    adv_image[y1:y2, x1:x2] = adv_face

    return adv_image