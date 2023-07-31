import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import random



def add_noise(tensor, poisson_rate, gaussian_std_dev,):
    gaussian_noise = gaussian_std_dev * torch.randn(tensor.size())
    poisson_noise = torch.poisson(torch.full(tensor.size(), poisson_rate))
    noisy_tensor = tensor + gaussian_noise + poisson_noise
    return noisy_tensor

def plot_images(original_images, noisy_images):
    num_images = len(original_images)
    fig, axes = plt.subplots(num_images, 2, figsize=(6, 3 * num_images))
    for i in range(num_images):
        # Convert tensors to NumPy arrays and transpose dimensions
        original_np = original_images[i].permute(1, 2, 0).numpy()
        noisy_np = noisy_images[i].permute(1, 2, 0).numpy()

        # Plot original image
        axes[i, 0].imshow(original_np)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # Plot noisy image
        axes[i, 1].imshow(noisy_np)
        axes[i, 1].set_title('Noisy Image')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()





# Load and plot original and noisy images
def load_and_plot_images(data_path, num_images=5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # train_dataset = torchvision.datasets.CIFAR10(root='D:/Research/Super/autoencoder/dataset/', download=True,
    #                                              transform=train_transform)
    custom_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    indices = torch.randint(0, len(custom_dataset), (num_images,)).tolist()

    original_images = torch.stack([custom_dataset[idx][0] for idx in indices], dim=0)


    poisson_rate = random.uniform(0.2, 0.4)
    gaussian_std_dev = random.uniform(0.2,0.4)
    # salt_pepper_prob = random.uniform(0.01, 0.10)
    noisy_images = add_noise(original_images, poisson_rate, gaussian_std_dev)

    # noisy_images = original_images + 0.2 * torch.randn(original_images.size())

    plot_images(original_images, noisy_images)


# Load and plot images from custom dataset
load_and_plot_images('D:/Research/Super/dataset/original/VEDAI_1024/New folder/', num_images=5)
