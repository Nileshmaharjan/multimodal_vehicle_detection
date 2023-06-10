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

def add_lognormal_noise_and_gaussian_kernel(image, mean, std):
    """
    Adds Log-Normal noise to an image.
    """
    # Generate the noise array
    noise_array = np.random.lognormal(mean, std, image.shape)

    # Add the noise to the image array
    noisy_image_array = image.astype(np.float32) + noise_array

    # Clip the noisy image array to the valid range [0, 255]
    noisy_image_array = np.clip(noisy_image_array, 0, 255)

    # Convert the noisy image array back to an image
    noisy_image = noisy_image_array.astype(np.float32)

    blurred_array = gaussian_filter(noisy_image, sigma=1)

    return blurred_array


# Define the denoising autoencoder model
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_model():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
    num_epochs = 200
    batch_size = 16
    learning_rate = 0.001
    early_stopping_epochs = 10
    early_stopping_tol = 1e-4

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Custom dataset and data loader
    train_dataset = torchvision.datasets.ImageFolder(root='D:/Research/Super/dataset/original/VEDAI_1024/images/', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model and move it to GPU if available
    model = DenoisingAutoencoder().to(device)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # TensorBoard writer
    writer = SummaryWriter()

    # Training loop
    total_step = len(train_loader)
    best_loss = float('inf')
    no_improvement_count = 0
    print('Training started')
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(train_loader):

            # Move images to GPU if available
            images = images.to(device)
            # Add noise to the input images

            # noisy_images = images + 0.2 * torch.randn(images.size(), device=device)

            # Convert the list of NumPy arrays to a single NumPy array
            noisy_images_array = np.array([add_lognormal_noise_and_gaussian_kernel(img.cpu().numpy(), 0.2, 0.2) for img in images])

            # Convert the NumPy array to a tensor
            noisy_images = torch.tensor(noisy_images_array)
            noisy_images = noisy_images.to(device)

            # print('here', noisy_images)

            # Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if (i + 1) % 100 == 0:
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)



        # Save the model checkpoint if validation loss improves
        if loss.item() < best_loss - early_stopping_tol:
            best_loss = loss.item()
            no_improvement_count = 0
            torch.save(model.state_dict(), f'denoising_autoencoder_best.ckpt')
        else:
            no_improvement_count += 1
        # Check early stopping condition

        print('Epoch:', epoch, 'Validation loss: ', loss.item())
        if no_improvement_count >= early_stopping_epochs:
            print("Early stopping. No improvement in validation loss.")
            break

    writer.close()
    print("Training finished.")


# Function to plot original and noisy images
def plot_images(original_images, noisy_images):
    num_images = len(original_images)
    fig, axes = plt.subplots(num_images, 2, figsize=(6, 3 * num_images))
    for i in range(num_images):
        # Plot original image
        axes[i, 0].imshow(original_images[i].permute(1, 2, 0))
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # Plot noisy image
        axes[i, 1].imshow(noisy_images[i].permute(1, 2, 0))
        axes[i, 1].set_title('Noisy Image')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


# Run the training
train_model()

# Load and plot original and noisy images
def load_and_plot_images(data_path, num_images=5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    custom_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    indices = torch.randint(0, len(custom_dataset), (num_images,)).tolist()

    original_images = torch.stack([custom_dataset[idx][0] for idx in indices], dim=0)
    noisy_images = original_images + 0.2 * torch.randn(original_images.size())

    plot_images(original_images, noisy_images)


# Load and plot images from custom dataset
load_and_plot_images('path_to_image_directory', num_images=5)
