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



def generate_adversarial(image, epsilon=0.012, steps=10, alpha=0.01):

    face_box = detect_face(image)

    if face_box is None:
        x1, y1 = 0, 0
        x2, y2 = image.shape[1], image.shape[0]
        face = image
    else:
        x1, y1, x2, y2 = face_box
        face = image[y1:y2, x1:x2]

    face = cv2.resize(face,(160,160))

    tensor = transform(face).unsqueeze(0)
    original = tensor.clone().detach()

    original_embedding = model(original).detach()

    for _ in range(steps):

        tensor.requires_grad = True

        embedding = model(tensor)

        loss = -torch.nn.functional.cosine_similarity(embedding,original_embedding).mean()

        #loss = torch.nn.functional.mse_loss(embedding, original_embedding)

        loss.backward()

        perturbation = alpha * tensor.grad.sign()

        tensor = tensor + perturbation

        # project perturbation within epsilon-ball
        delta = torch.clamp(tensor - original, -epsilon, epsilon)
        tensor = torch.clamp(original + delta, 0, 1).detach()

    adv_face = tensor.squeeze().permute(1,2,0).numpy()
    adv_face = (adv_face * 255).astype(np.uint8)

    adv_face = cv2.resize(adv_face,(x2-x1,y2-y1))

    adv_image = image.copy()
    adv_image[y1:y2,x1:x2] = adv_face

    return adv_image
