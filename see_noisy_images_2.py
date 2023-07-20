import cv2
import torch
import numpy as np
import random

def add_noise(image, poisson_rate, gaussian_std_dev):
    # Convert the image to NumPy array
    noisy_np = image.numpy()

    # Add Gaussian noise
    gaussian_noise = np.random.normal(scale=gaussian_std_dev, size=noisy_np.shape)
    noisy_np += gaussian_noise

    # # Add Poisson noise
    poisson_noise = np.random.poisson(lam=poisson_rate, size=noisy_np.shape)
    noisy_np += poisson_noise

    # Clip values to [0, 255] range
    noisy_np = np.clip(noisy_np, 0, 255)

    # Convert NumPy array back to tensor
    noisy_image = torch.from_numpy(noisy_np)

    return noisy_image

# Read the image using cv2
image_path = 'data/images/00000000_co.png'  # Replace with the actual image path
image = cv2.imread(image_path)

# Convert the BGR image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to a PyTorch tensor
image_tensor = torch.from_numpy(image_rgb.transpose((2, 0, 1))).float() / 255.0

# Add noise to the image
poisson_rate = random.uniform(0.1, 0.2)
gaussian_std_dev = random.uniform(0.01, 0.05)

noisy_image_tensor = add_noise(image_tensor, poisson_rate, gaussian_std_dev)

# Convert the noisy image back to a NumPy array for visualization
noisy_image_np = (noisy_image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

# Display the original and noisy images
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
