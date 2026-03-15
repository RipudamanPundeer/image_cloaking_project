import os
import numpy as np

from facenet_pytorch import InceptionResnetV1

from preprocess import load_image
from perturbation import generate_adversarial
from evaluation import cosine_similarity


IMAGE_FOLDER = "images"
EPSILON = 0.03


# Model used to generate perturbation
model_attack = InceptionResnetV1(pretrained='vggface2').eval()

# Second model used for evaluation
model_test = InceptionResnetV1(pretrained='casia-webface').eval()


def get_embedding(model, image):

    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(tensor)

    return embedding


images = os.listdir(IMAGE_FOLDER)

similarities = []

print("\nTransferability Test\n")


for img in images:

    path = os.path.join(IMAGE_FOLDER,img)

    _, image = load_image(path)

    protected = generate_adversarial(image, EPSILON)


    emb_original = get_embedding(model_test, image)

    emb_protected = get_embedding(model_test, protected)


    similarity = cosine_similarity(
        emb_original,
        emb_protected
    ).item()

    similarities.append(similarity)

    print(f"{img} -> similarity (transfer test): {similarity:.3f}")


print("\nAverage similarity:", np.mean(similarities))