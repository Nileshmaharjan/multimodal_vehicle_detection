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
    # transforms.Normalize((0.46961793,0.44640928, 0.40719114), (0.23942938, 0.23447396, 0.23768907))
])

# Load the test dataset
data_path_test = r"C:/Users/User/Documents/Projects/Nilesh/fso_traffic_surveillance/autoencoder/images/test/"
test_dataset = torchvision.datasets.ImageFolder(root=data_path_test, transform=test_transform)

# Create a DataLoader for the test dataset
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the saved checkpoint
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
checkpoint = torch.load(checkpoint_path, map_location=device)

import torch
import torch.nn.functional as F
import torch.nn as nn
import configs.models_config as config


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
def visualize_denoised_images(original, noisy, denoised, num_images=6):
    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3*num_images))
    for i in range(num_images):
        axes[i, 0].imshow(noisy[i].permute(1, 2, 0))
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(original[i].permute(1, 2, 0))
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(denoised[i].permute(1, 2, 0))
        axes[i, 2].set_title('Denoised')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

# Select a batch of test images
test_images, _ = next(iter(test_loader))

# Add noise to the test images
poisson_rate = 0.1
gaussian_std_dev = 0.05
noisy_images = add_noise(test_images, poisson_rate, gaussian_std_dev)
noisy_images = noisy_images.to(device)
print('noisy images')

# Apply the denoiser model to the noisy images
with torch.no_grad():
    denoised_images = autoencoder(noisy_images)

# Visualize the original, noisy, and denoised images
visualize_denoised_images(test_images, noisy_images.cpu(), denoised_images.cpu())


# def load_ir(self, index, denoising_model): #zjq
#     # loads 1 image from dataset, returns img, original hw, resized hw
#     ir = self.irs[index]
#     if ir is None:  # not cached
#         path = self.ir_files[index]
#         ir = cv2.imread(path)  # BGR
#
#         # Convert the image to a PyTorch tensor
#         image_tensor = torch.from_numpy(ir.transpose((2, 0, 1))).float() / 255.0
#
#         # Add noise
#         poisson_rate = random.uniform(0.1, 0.2)
#         gaussian_std_dev = random.uniform(0.1, 0.2)
#
#         noisy_image_tensor = self.add_noise(image_tensor, poisson_rate, gaussian_std_dev)
#
#         # Denoise the image using the provided denoising_model
#         with torch.no_grad():
#             denoised_image_tensor = denoising_model(noisy_image_tensor.unsqueeze(0))
#
#         # Convert the denoised image back to a NumPy array for visualization
#         denoised_image = (denoised_image_tensor.squeeze().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#
#
#         assert ir is not None, 'Image_ir Not Found ' + path
#         h0, w0 = ir.shape[:2]  # orig hw
#         r = self.img_size / max(h0, w0)  # resize image to img_size
#         if r != 1:  # always resize down, only resize up if training with augmentation
#             interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
#             denoised_image  = cv2.resize(denoised_image, (int(w0 * r), int(h0 * r)), interpolation=interp)
#         return denoised_image    # denoised_image   ##, hw_original, hw_resized
#     else:
#         return self.irs[index]  # img ##, hw_original, hw_resized
#
# def load_ir(self, index, denoising_model): #zjq
#     # loads 1 image from dataset, returns img, original hw, resized hw
#     ir = self.irs[index]
#     if ir is None:  # not cached
#         path = self.ir_files[index]
#         ir = cv2.imread(path)  # BGR
#
#         # Add noise
#
#         # Convert the image to a PyTorch tensor
#         image_tensor = torch.from_numpy(ir.transpose((2, 0, 1))).float() / 255.0
#
#         # Add noise to the image
#         poisson_rate = random.uniform(0.1, 0.2)
#         gaussian_std_dev = random.uniform(0.01, 0.05)
#
#         noisy_image_tensor = add_noise(image_tensor, poisson_rate, gaussian_std_dev)
#
#         # Convert the noisy image back to a NumPy array for visualization
#         ir = (noisy_image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#
#         assert ir is not None, 'Image_ir Not Found ' + path
#         h0, w0 = ir.shape[:2]  # orig hw
#         r = self.img_size / max(h0, w0)  # resize image to img_size
#         if r != 1:  # always resize down, only resize up if training with augmentation
#             interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
#             ir = cv2.resize(ir, (int(w0 * r), int(h0 * r)), interpolation=interp)
#         return ir  # ir ##, hw_original, hw_resized
#     else:
#         return self.irs[index]  # img ##, hw_original, hw_resized
