import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch import nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Define the path to the saved checkpoint
checkpoint_path = "C:/Users/User/Documents/Projects/Nilesh/fso_traffic_surveillance/autoencoder/checkpoint/model_checkpoint_epoch_9.pt"

# Define the transformation for the test data
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

class EncoderBlock(nn.Module):
    """CNN-based encoder block"""

    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=config.kernel_size,
                              padding=config.padding, stride=config.stride, bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

    def forward(self, ten, out=False, t=False):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten, True)
        return ten

class DecoderBlock(nn.Module):
    """CNN-based decoder block"""

    def __init__(self, channel_in, channel_out, out=False):
        super(DecoderBlock, self).__init__()

        if out:
            self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=config.kernel_size, padding=config.padding,
                                           stride=config.stride, output_padding=1, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=config.kernel_size,
                                           padding=config.padding, stride=config.stride, bias=False)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten, True)
        return ten

class Encoder(nn.Module):
    """VAE-based encoder"""

    def __init__(self, channel_in=3, z_size=128):
        super(Encoder, self).__init__()

        self.size = channel_in
        layers_list = []
        for i in range(3):
            layers_list.append(EncoderBlock(channel_in=self.size, channel_out=config.encoder_channels[i]))
            self.size = config.encoder_channels[i]
        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(nn.Linear(in_features=config.fc_input * config.fc_input * self.size,
                                          out_features=config.fc_output, bias=False),
                                nn.BatchNorm1d(num_features=config.fc_output, momentum=0.9),
                                nn.ReLU(True))
        self.l_mu = nn.Linear(in_features=config.fc_output, out_features=z_size)
        self.l_var = nn.Linear(in_features=config.fc_output, out_features=z_size)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        mu = self.l_mu(ten)
        logvar = self.l_var(ten)
        return mu, logvar

class Decoder(nn.Module):
    """VAE-based decoder"""

    def __init__(self, z_size, size):
        super(Decoder, self).__init__()

        self.fc = nn.Sequential(nn.Linear(in_features=z_size, out_features=config.fc_input * config.fc_input * size, bias=False),
                                nn.BatchNorm1d(num_features=config.fc_input * config.fc_input * size, momentum=0.9),
                                nn.ReLU(True))
        self.size = size
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size, out=config.output_pad_dec[0]))
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=config.decoder_channels[1], out=config.output_pad_dec[1]))
        self.size = config.decoder_channels[1]
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=config.decoder_channels[2], out=config.output_pad_dec[2]))
        self.size = config.decoder_channels[2]
        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=config.decoder_channels[3], kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        ))
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.fc(ten)
        ten = ten.view(len(ten), -1, config.fc_input, config.fc_input)
        ten = self.conv(ten)
        return ten

class Autoencoder(nn.Module):
    """VAE model: encoder + decoder + re-parametrization layer"""

    def __init__(self, device, z_size=128):
        super(Autoencoder, self).__init__()

        self.z_size = z_size  # latent space size
        self.encoder = Encoder(z_size=self.z_size).to(device)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size).to(device)
        self.init_parameters()
        self.device = device

    def init_parameters(self):

        """Glorot initialization"""

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    nn.init.xavier_normal_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias, 0.0)

    def reparametrize(self, mu, logvar):

        """ Re-parametrization trick"""

        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())

        return eps.mul(logvar).add_(mu)

    def forward(self, x, gen_size=10):

        if x is not None:
            x = Variable(x).to(self.device)

        if self.training:
            mus, log_variances = self.encoder(x)
            z = self.reparametrize(mus, log_variances)
            x_tilde = self.decoder(z)

            # generate from random latent variable
            z_p = Variable(torch.randn(len(x), self.z_size).to(self.device), requires_grad=True)
            x_p = self.decoder(z_p)
            return x_tilde, x_p, mus, log_variances, z_p

        else:
            if x is None:
                z_p = Variable(torch.randn(gen_size, self.z_size).to(self.device), requires_grad=False)
                x_p = self.decoder(z_p)
                return x_p

            else:
                mus, log_variances = self.encoder(x)
                z = self.reparametrize(mus, log_variances)
                x_tilde = self.decoder(z)
                return x_tilde

    def __call__(self, *args, **kwargs):
        return super(Autoencoder, self).__call__(*args, **kwargs)

# Instantiate the Autoencoder
autoencoder = Autoencoder(device)
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
def visualize_denoised_images(original, noisy, denoised, num_images=5):
    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3*num_images))
    for i in range(num_images):
        axes[i, 0].imshow(original[i].permute(1, 2, 0))
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(noisy[i].permute(1, 2, 0))
        axes[i, 1].set_title('Noisy')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(denoised[i].permute(1, 2, 0))
        axes[i, 2].set_title('Denoised')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

# Select a batch of test images
test_images, _ = next(iter(test_loader))

# Add noise to the test images
poisson_rate = 0.02
gaussian_std_dev = 0.02
noisy_images = add_noise(test_images, poisson_rate, gaussian_std_dev)
noisy_images = noisy_images.to(device)

# Apply the denoiser model to the noisy images
with torch.no_grad():
    denoised_images = autoencoder(noisy_images)

# Visualize the original, noisy, and denoised images
visualize_denoised_images(test_images, noisy_images.cpu(), denoised_images.cpu())
