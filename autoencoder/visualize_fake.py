import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch import nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Define the path to the saved checkpoint
checkpoint_path = "C:/Users/User/Documents/Projects/Nilesh/fso_traffic_surveillance/autoencoder/checkpoint-unormalized-coo/model_checkpoint_epoch_28.pt"


# Define the transformation for the test data
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load the test dataset
data_path_test = r"C:/Users/User/Documents/Projects/Nilesh/fso_traffic_surveillance/autoencoder/images/test/"
test_dataset = torchvision.datasets.ImageFolder(root=data_path_test, transform=test_transform)

# Create a DataLoader for the test dataset
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load the saved checkpoint
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
checkpoint = torch.load(checkpoint_path, map_location=device)

import torch
import torch.nn.functional as F
import torch.nn as nn
import configs.models_config as config
import cv2


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.conv5a = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.conv5b = nn.Conv2d(48, 48, kernel_size=3, padding=1)

        # Decoder
        self.conv6a = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv6b = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv7a = nn.Conv2d(144, 96, kernel_size=3, padding=1)
        self.conv7b = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv8a = nn.Conv2d(144, 96, kernel_size=3, padding=1)
        self.conv8b = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv9a = nn.Conv2d(144, 96, kernel_size=3, padding=1)
        self.conv9b = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv10a = nn.Conv2d(99, 64, kernel_size=3, padding=1)
        self.conv10b = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv11 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Encoder
        x1 = F.leaky_relu(self.conv1(x))
        x1 = F.leaky_relu(self.conv1b(x1))
        x2 = F.max_pool2d(x1, kernel_size=2, stride=2)
        x2 = F.leaky_relu(self.conv2(x2))
        x3 = F.max_pool2d(x2, kernel_size=2, stride=2)
        x3 = F.leaky_relu(self.conv3(x3))
        x4 = F.max_pool2d(x3, kernel_size=2, stride=2)
        x4 = F.leaky_relu(self.conv4(x4))
        x5 = F.max_pool2d(x4, kernel_size=2, stride=2)
        x5 = F.leaky_relu(self.conv5a(x5))
        x5_2 = F.max_pool2d(x5, kernel_size=2, stride=2)
        x5_2 = F.leaky_relu(self.conv5b(x5_2))

        # Decoder
        x6 = self.upsample(x5_2)
        x6 = torch.cat((x6, x5), dim=1)
        x6 = F.leaky_relu(self.conv6a(x6))
        x6 = F.leaky_relu(self.conv6b(x6))
        x7 = self.upsample(x6)
        x7 = torch.cat((x7, x4), dim=1)
        x7 = F.leaky_relu(self.conv7a(x7))
        x7 = F.leaky_relu(self.conv7b(x7))
        x8 = self.upsample(x7)
        x8 = torch.cat((x8, x3), dim=1)
        x8 = F.leaky_relu(self.conv8a(x8))
        x8 = F.leaky_relu(self.conv8b(x8))
        x9 = self.upsample(x8)
        x9 = torch.cat((x9, x2), dim=1)
        x9 = F.leaky_relu(self.conv9a(x9))
        x9 = F.leaky_relu(self.conv9b(x9))
        x10 = self.upsample(x9)
        x10 = torch.cat((x10, x), dim=1)
        x10 = F.leaky_relu(self.conv10a(x10))
        x10 = F.leaky_relu(self.conv10b(x10))
        decoded = self.sigmoid(self.conv11(x10))

        return decoded


# Instantiate the Autoencoder
autoencoder = Autoencoder()
autoencoder.load_state_dict(checkpoint['model_state_dict'])
autoencoder.to(device)
autoencoder.eval()

# Function to add noise to images
def add_noise(tensor, poisson_rate, gaussian_std_dev):
    gaussian_noise = gaussian_std_dev * torch.randn(tensor.size())
    poisson_noise = torch.poisson(torch.full(tensor.size(), poisson_rate))
    noisy_tensor = tensor + gaussian_noise + poisson_noise
    noisy = torch.clip(noisy_tensor, 0., 1.)
    return noisy

# Function to display the original, noisy, and denoised images

def resize_image(image, target_size):
    return cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
def visualize_denoised_images(original, noisy, denoised, num_images=5):
    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3*num_images))
    for i in range(num_images):
        resized_noisy = resize_image(noisy[i].permute(1, 2, 0).numpy(), (1024, 1024))
        print('here')
        axes[i, 0].imshow(resized_noisy)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')

        resized_original = resize_image(original[i].permute(1, 2, 0).numpy(), (1024, 1024))
        axes[i, 1].imshow(resized_original)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        resized_denoised = resize_image(denoised[i].permute(1, 2, 0).numpy(), (1024, 1024))
        axes[i, 2].imshow(resized_denoised)
        axes[i, 2].set_title('Denoised')
        axes[i, 2].axis('off')

        # # Now you can plot the images using Matplotlib
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(resized_original)
        # plt.axis('off')
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(resized_original)
        # plt.axis('off')
        #
        #
        #
        # # Save the plot as an image file (e.g., PNG or JPEG)
        # plt.savefig('original.png', bbox_inches='tight', pad_inches=0.1)

    plt.tight_layout()
    plt.show()

# Select a batch of test images
test_images, _ = next(iter(test_loader))

# Add noise to the test images
poisson_rate = 2.00
gaussian_std_dev = 0.01
noisy_images = add_noise(test_images, poisson_rate, gaussian_std_dev)
noisy_images = noisy_images.to(device)
print('here')

# Apply the denoiser model to the noisy images
with torch.no_grad():
    denoised_images = autoencoder(noisy_images)
    print('here')

# Visualize the original, noisy, and denoised images
visualize_denoised_images(test_images, noisy_images.cpu(), denoised_images.cpu())



