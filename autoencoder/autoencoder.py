import numpy as np  # this module is useful to work with numerical arrays
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import random
from tqdm import tqdm  # Import tqdm for the progress bar
import os
import torch, gc
gc.collect()
torch.cuda.empty_cache()
from torch.autograd import Variable

# Set the directory for TensorBoard logs
log_dir = r"C:/Users/User/Documents/Projects/Nilesh/fso_traffic_surveillance/autoencoder/runs/"


# Create a SummaryWriter object
writer = SummaryWriter(log_dir=log_dir)

data_path_1 = r"C:/Users/User/Documents/Projects/Nilesh/fso_traffic_surveillance/autoencoder/images/color/"
data_path_2 = r"C:/Users/User/Documents/Projects/Nilesh/fso_traffic_surveillance/autoencoder/images/ir/"
data_path_3 = r"C:/Users/User/Documents/Projects/Nilesh/fso_traffic_surveillance/autoencoder/images/coco/"


# train_transform1 = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])
#
# train_transform2 = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])

train_transform3 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# train_dataset1 = torchvision.datasets.ImageFolder(root=data_path_1, transform=train_transform1)
# train_dataset2 = torchvision.datasets.ImageFolder(root=data_path_2, transform=train_transform2)
train_dataset3 = torchvision.datasets.ImageFolder(root=data_path_3, transform=train_transform3)

# # Define the length of each dataset
# m1 = len(train_dataset1)
# m2 = len(train_dataset2)
m3 = len(train_dataset3)

# Split the datasets
# train_data1, val_data1 = random_split(train_dataset1, [int(0.75*m1), int(0.25*m1)])
# train_data2, val_data2 = random_split(train_dataset2, [int(0.75*m2), int(0.25*m2)])
train_data3, val_data3 = random_split(train_dataset3, [82800,35487])

# # Combine the train datasets
# combined_train_data = ConcatDataset([train_data1, train_data2,train_data3])
# combined_val_data = ConcatDataset([val_data1, val_data2,val_data3])

batch_size = 256

train_loader = torch.utils.data.DataLoader(train_data3, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data3, batch_size=batch_size)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

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


# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('Cuda available:', torch.cuda.is_available())
print(f'Selected device: {device}')

# Instantiate the Autoencoder
autoencoder = Autoencoder(device)

# Define the loss function
loss_fn = torch.nn.MSELoss()

# Define an optimizer
lr = 0.001
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=1e-07)



# Move the autoencoder to the selected device
autoencoder.to(device)

# Function to save model checkpoint

checkpoint_path = "C:/Users/User/Documents/Projects/Nilesh/fso_traffic_surveillance/autoencoder/checkpoint/model_checkpoint_epoch_10.pt"
def save_checkpoint(epoch, model, optimizer, loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    checkpoint_path = f"C:/Users/User/Documents/Projects/Nilesh/fso_traffic_surveillance/autoencoder/checkpoint/model_checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved at epoch {epoch}!")


def add_noise(tensor, poisson_rate, gaussian_std_dev):
    gaussian_noise = gaussian_std_dev * torch.randn(tensor.size())
    poisson_noise = torch.poisson(torch.full(tensor.size(), poisson_rate))
    noisy_tensor = tensor + gaussian_noise + poisson_noise
    noisy = torch.clip(noisy_tensor, 0., 1.)
    return noisy


def train_epoch_den(autoencoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for the autoencoder
    autoencoder.train()
    train_loss = []

    # Iterate over the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader:  # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensors to the proper device
        poisson_rate = random.uniform(0.01, 0.02)
        gaussian_std_dev = random.uniform(0.01, 0.02)


        image_noisy = add_noise(image_batch, poisson_rate, gaussian_std_dev)
        image_batch = image_batch.to(device)
        image_noisy = image_noisy.to(device)
        # Encode and Decode data
        decoded_data = autoencoder(image_noisy)
        decoded_data = decoded_data[0].to(device)


        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.item())

    return np.mean(train_loss)

def test_epoch_den(autoencoder, device, dataloader, loss_fn):
    # Set evaluation mode for the autoencoder
    autoencoder.eval()


    with torch.no_grad():  # No need to track the gradients
        val_loss_list = []
        for image_batch, _ in dataloader:
            # Move tensors to the proper device
            poisson_rate = random.uniform(0.01, 0.02)
            gaussian_std_dev = random.uniform(0.01, 0.02)

            image_noisy = add_noise(image_batch, poisson_rate, gaussian_std_dev)
            image_noisy = image_noisy.to(device)
            # Encode and Decode data
            image_batch = image_batch.to(device)
            decoded_data = autoencoder(image_noisy)
            decoded_data = decoded_data.to(device)

            # Evaluate loss
            val_loss = loss_fn(decoded_data, image_batch)

            val_loss_list.append(val_loss.item())

    return np.mean(val_loss_list)


# Load checkpoint if it exists
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    start_epoch, val_loss = load_checkpoint(autoencoder, optimizer, checkpoint_path)
    print(f"Checkpoint loaded from epoch {start_epoch}. Resuming training...")
else:
    start_epoch = 0
    val_loss = float('inf')


### Training cycle
noise_factor = 0.3
num_epochs = 100
history_da = {'train_loss': [], 'val_loss': []}
patience = 20


for epoch in range(start_epoch, num_epochs):

    torch.cuda.empty_cache()
    print('EPOCH %d/%d' % (epoch + 1, num_epochs))

    # Initialize tqdm for the training data loader
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)")

    ### Training (use the training function)
    train_loss  = train_epoch_den(
        autoencoder=autoencoder,
        device=device,
        dataloader=train_loader_tqdm,
        loss_fn=loss_fn,
        optimizer=optimizer)
    print('train_loss', train_loss)
    # Close the tqdm progress bar for the training data loader
    train_loader_tqdm.close()

    # Initialize tqdm for the validation data loader
    valid_loader_tqdm = tqdm(valid_loader, desc=f"Epoch {epoch + 1} (Validation)")

    ### Validation (use the testing function)
    val_loss = test_epoch_den(
        autoencoder=autoencoder,
        device=device,
        dataloader=valid_loader_tqdm,
        loss_fn=loss_fn)

    # Close the tqdm progress bar for the validation data loader
    valid_loader_tqdm.close()

    # Write train_loss and val_loss to TensorBoard
    writer.add_scalar('Train Loss', train_loss, epoch)
    writer.add_scalar('Validation Loss', val_loss, epoch)


    save_checkpoint(epoch, autoencoder, optimizer, val_loss)

    # Print Validation loss
    history_da['train_loss'].append(train_loss)
    history_da['val_loss'].append(val_loss)

    print('\n EPOCH {}/{} \t train loss {:.5f} \t val loss {:.5f}'.format(
        epoch + 1, num_epochs, train_loss, val_loss))

