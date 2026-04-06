import sys
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim

# Import your existing, untouched modules
from evaluation import model
from face_detection import detect_face

transform = transforms.Compose([
    transforms.ToTensor()
])


def diagnose_adversarial(image, epsilon=0.05, steps=10, alpha=0.01):
    print("\n" + "=" * 50)
    print("🚀 STARTING ADVERSARIAL DIAGNOSTICS")
    print("=" * 50)

    device = next(model.parameters()).device

    # [DIAGNOSTIC 3] Coordinate Clipping & Edge Cases
    print(f"\n[DIAGNOSTIC 3] Image Shape: {image.shape}")
    face_box = detect_face(image)

    if face_box is None:
        print("❌ FAILED: No face detected.")
        return image

    print(f"[DIAGNOSTIC 3] Raw MTCNN Box: {face_box}")
    x1, y1, x2, y2 = face_box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    print(f"[DIAGNOSTIC 3] Clamped Box:   [{x1}, {y1}, {x2}, {y2}]")

    # [DIAGNOSTIC 1] Resolution Discrepancy
    box_width, box_height = x2 - x1, y2 - y1
    print(f"\n[DIAGNOSTIC 1] Native Face Size: {box_width}x{box_height}")
    if box_width > 300 or box_width < 100:
        print("⚠️ WARNING: High risk of interpolation destroying noise during re-sizing.")

    face = image[y1:y2, x1:x2]
    face_resized = cv2.resize(face, (160, 160))
    tensor = transform(face_resized).unsqueeze(0).to(device)
    original = tensor.clone().detach()

    with torch.no_grad():
        original_embedding = model(original).detach()

    adv = original.clone().detach()
    momentum = torch.zeros_like(original)
    mu = 1.0

    print("\n[DIAGNOSTIC 4] Tracking Gradients in PGD Loop...")
    for step in range(steps):
        adv.requires_grad = True
        embedding = model(adv)

        loss = F.cosine_similarity(embedding, original_embedding).mean()
        grad = torch.autograd.grad(loss, adv)[0]

        # [DIAGNOSTIC 4] Zero Gradients
        max_grad = torch.max(torch.abs(grad)).item()
        if step == 0 or step == steps - 1:
            print(f"  -> Step {step + 1}/{steps} | Loss: {loss.item():.4f} | Max Grad: {max_grad:.8f}")
            if max_grad < 1e-7:
                print("  ❌ WARNING: Gradient is essentially zero. The model is too confident.")

        grad_norm = torch.norm(grad, p=1)
        grad_normalized = grad / (grad_norm + 1e-8)
        momentum = mu * momentum + grad_normalized

        adv = adv.detach() - alpha * momentum.sign()
        delta = torch.clamp(adv - original, -epsilon, epsilon)
        adv = torch.clamp(original + delta, 0.0, 1.0).detach()

    # Calculate Tensor-level SSIM before OpenCV gets involved
    tensor_adv_np = adv.squeeze().permute(1, 2, 0).cpu().numpy()
    tensor_orig_np = original.squeeze().permute(1, 2, 0).cpu().numpy()
    tensor_ssim = ssim(tensor_orig_np, tensor_adv_np, channel_axis=-1, data_range=1.0)
    print(f"\n[MATH VERIFICATION] SSIM of pure 160x160 tensors: {tensor_ssim:.4f}")

    # [DIAGNOSTIC 2] Type-Casting and Re-stitching
    adv_face = (tensor_adv_np * 255).astype(np.uint8)
    adv_face_resized = cv2.resize(adv_face, (box_width, box_height))

    print(f"\n[DIAGNOSTIC 2] Paste Validation")
    print(
        f"  -> adv_face_resized dtype: {adv_face_resized.dtype}, Min: {adv_face_resized.min()}, Max: {adv_face_resized.max()}")

    target_patch = image[y1:y2, x1:x2]
    print(f"  -> target_patch dtype:     {target_patch.dtype}, Min: {target_patch.min()}, Max: {target_patch.max()}")

    adv_image = image.copy()
    adv_image[y1:y2, x1:x2] = adv_face_resized

    # Final sanity check: Did the paste actually change the image array?
    final_patch = adv_image[y1:y2, x1:x2]
    patch_ssim = ssim(target_patch, final_patch, channel_axis=-1)

    print(f"\n[FINAL VERIFICATION] SSIM between original crop and pasted crop: {patch_ssim:.4f}")
    if patch_ssim > 0.99:
        print("❌ FATAL: The pasted face is mathematically identical to the original.")
        print("          Interpolation completely destroyed the noise during cv2.resize().")
    else:
        print("✅ SUCCESS: The adversarial noise survived the re-stitching process.")

    print("=" * 50 + "\n")
    return adv_image


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_runner.py <path_to_failing_image.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image at {image_path}. Check the path.")
        sys.exit(1)

    # Convert BGR to RGB (Standard for PyTorch models)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run the diagnostics
    adv_image_rgb = diagnose_adversarial(image_rgb)

    # Convert back to BGR for OpenCV to save properly
    adv_image_bgr = cv2.cvtColor(adv_image_rgb, cv2.COLOR_RGB2BGR)

    output_filename = "debug_output.jpg"
    cv2.imwrite(output_filename, adv_image_bgr)
    print(f"💾 Saved the final perturbed image to {output_filename}")