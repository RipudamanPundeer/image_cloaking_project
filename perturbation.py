import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
import numpy as np

model = InceptionResnetV1(pretrained='vggface2').eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

def generate_adversarial(image, epsilon=0.01):

    tensor = transform(image).unsqueeze(0)
    tensor.requires_grad = True

    embedding = model(tensor)

    loss = embedding.norm()
    loss.backward()

    perturbation = epsilon * tensor.grad.sign()

    adv_image = tensor + perturbation
    adv_image = torch.clamp(adv_image, 0, 1)

    adv_image = adv_image.squeeze().permute(1,2,0).detach().numpy()
    adv_image = (adv_image * 255).astype(np.uint8)

    return adv_image