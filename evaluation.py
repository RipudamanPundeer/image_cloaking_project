import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from skimage.metrics import structural_similarity as ssim
import numpy as np

model = InceptionResnetV1(pretrained='vggface2').eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

def get_embedding(image):
    tensor = transform(image).unsqueeze(0)
    embedding = model(tensor)
    return embedding

def cosine_similarity(e1, e2):
    return torch.nn.functional.cosine_similarity(e1, e2)

def compute_ssim(img1, img2):
    score = ssim(img1, img2, channel_axis=2)
    return score