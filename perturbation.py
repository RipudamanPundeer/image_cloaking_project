import numpy as np

def generate_perturbation(image, epsilon=5):
    noise = np.random.randint(-epsilon, epsilon, image.shape)
    perturbed = image + noise
    perturbed = np.clip(perturbed, 0, 255)
    return perturbed.astype('uint8')