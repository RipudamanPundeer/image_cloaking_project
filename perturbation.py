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

def generate_adversarial(image, epsilon=0.02, steps=10, alpha=0.007):

    device = next(model.parameters()).device


    # ---------------------------
    # FACE DETECTION
    # ---------------------------

    face_box = detect_face(image)

    if face_box is None:

        x1, y1 = 0, 0
        x2, y2 = image.shape[1], image.shape[0]

        face = image

    else:

        x1, y1, x2, y2 = face_box

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        face = image[y1:y2, x1:x2]


    # ---------------------------
    # RESIZE FACE
    # ---------------------------

    face = cv2.resize(face, (160,160))

    tensor = transform(face).unsqueeze(0).to(device)

    original = tensor.clone().detach()


    # ---------------------------
    # CREATE SPATIAL MASK
    # ---------------------------

    h, w = 160, 160

    y, x = np.ogrid[:h, :w]

    center_x = w // 2
    center_y = h // 2

    dist = np.sqrt((x-center_x)**2 + (y-center_y)**2)

    mask = np.exp(-dist / 60)

    mask = mask / mask.max()

    mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float().to(device)


    # ---------------------------
    # ORIGINAL EMBEDDING
    # ---------------------------

    with torch.no_grad():

        original_embedding = model(original)

        # random target embedding
        target_embedding = torch.randn_like(original_embedding)

        target_embedding = F.normalize(target_embedding)


    adv = original.clone().detach()


    # ---------------------------
    # PGD LOOP
    # ---------------------------

    for _ in range(steps):

        adv = adv.clone().detach().requires_grad_(True)

        embedding = model(adv)


        # ---------------------------
        # TARGETED REPULSION LOSS
        # ---------------------------

        loss = (
            F.cosine_similarity(embedding, original_embedding)
            - F.cosine_similarity(embedding, target_embedding)
        ).mean()


        grad = torch.autograd.grad(loss, adv)[0]


        # ---------------------------
        # MULTI-SCALE GRADIENT
        # ---------------------------

        small = F.interpolate(
            adv.detach(),
            size=(80,80),
            mode="bilinear",
            align_corners=False
        ).requires_grad_(True)

        small_up = F.interpolate(
            small,
            size=(160,160),
            mode="bilinear",
            align_corners=False
        )

        emb_small = model(small_up)

        loss_small = (
            F.cosine_similarity(emb_small, original_embedding)
            - F.cosine_similarity(emb_small, target_embedding)
        ).mean()

        grad_small = torch.autograd.grad(loss_small, small)[0]

        grad_small = F.interpolate(
            grad_small,
            size=(160,160),
            mode="bilinear",
            align_corners=False
        )


        combined_grad = (grad + grad_small) / 2


        # ---------------------------
        # APPLY SPATIAL MASK
        # ---------------------------

        perturbation = alpha * combined_grad.sign() * mask_tensor

        adv = adv + perturbation


        # ---------------------------
        # EPSILON PROJECTION
        # ---------------------------

        delta = torch.clamp(adv - original, -epsilon, epsilon)

        adv = torch.clamp(original + delta, 0, 1).detach()


    # ---------------------------
    # RECONSTRUCT IMAGE
    # ---------------------------

    adv_face = adv.squeeze().permute(1,2,0).cpu().numpy()

    adv_face = (adv_face * 255).astype(np.uint8)

    adv_face = cv2.resize(adv_face, (x2-x1, y2-y1))


    adv_image = image.copy()

    adv_image[y1:y2, x1:x2] = adv_face


    return adv_image