import torch
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision import transforms

from evaluation import model
from face_detection import detect_face

# ---------------------------
# IMAGE TRANSFORM
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])


# ---------------------------
# ADVERSARIAL GENERATOR
# ---------------------------
def generate_adversarial(image, epsilon=0.03, steps=10, alpha=0.01):
    device = next(model.parameters()).device

    # 1. Face Detection
    face_box = detect_face(image)
    if face_box is None:
        return image

    x1, y1, x2, y2 = face_box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    face = image[y1:y2, x1:x2]

    # 2. Resize and Prepare Tensor
    face_resized = cv2.resize(face, (160, 160))
    tensor = transform(face_resized).unsqueeze(0).to(device)
    original = tensor.clone().detach()

    # 3. Extract Original Embedding
    with torch.no_grad():
        original_embedding = model(original).detach()

    # 4. Streamlined & Aggressive PGD Loop
    adv = original.clone().detach()
    momentum = torch.zeros_like(original)  # ADDED: Initialize momentum
    mu = 1.0  # ADDED: Momentum decay factor

    for _ in range(steps):
        adv.requires_grad = True

        embedding = model(adv)

        # LOSS: We want to MINIMIZE cosine similarity to the original identity
        loss = F.cosine_similarity(embedding, original_embedding).mean()

        # Calculate gradient
        grad = torch.autograd.grad(loss, adv)[0]

        # --- ADDED: Momentum Calculation ---
        # Normalize the gradient to avoid vanishing/exploding gradients
        grad_norm = torch.norm(grad, p=1)
        grad_normalized = grad / (grad_norm + 1e-8)

        # Accumulate momentum
        momentum = mu * momentum + grad_normalized

        # GRADIENT DESCENT: Step in the negative direction to lower similarity
        adv = adv.detach() - alpha * grad.sign()

        # L-Infinity Projection (Keep noise within epsilon bounds)
        delta = torch.clamp(adv - original, -epsilon, epsilon)
        adv = torch.clamp(original + delta, 0.0, 1.0).detach()

    # 5. Reconstruct Image
    adv_face = adv.squeeze().permute(1, 2, 0).cpu().numpy()
    adv_face = (adv_face * 255).astype(np.uint8)

    # Resize perturbed face back to original bounding box size
    adv_face = cv2.resize(adv_face, (x2 - x1, y2 - y1))

    # Paste back into the original image
    adv_image = image.copy()
    adv_image[y1:y2, x1:x2] = adv_face

    return adv_image