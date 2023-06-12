import matplotlib.pyplot as plt  # plotting library
import numpy as np  # this module is useful to work with numerical arrays
import pandas as pd
import visualize_fake
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
# Set the directory for TensorBoard logs
log_dir = 'D:/Research/Super/autoencoder/runs/'

# Create a SummaryWriter object
writer = SummaryWriter(log_dir=log_dir)

data_dir = 'dataset'


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = torchvision.datasets.CIFAR10(root='D:/Research/Super/autoencoder/dataset/', download=True, transform=train_transform)
# train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)0
# test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)



# test_transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
#
# train_dataset.transform = train_transform
# test_dataset.transform = test_transform

m = len(train_dataset)
print('m',m)

train_data, val_data = random_split(train_dataset, [36000,14000])
batch_size = 64

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        ### Convolutional section
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )


    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.decoder(x)

        return x

### Define the loss function
loss_fn = torch.nn.MSELoss()

### Define an optimizer (both for the encoder and the decoder!)
lr = 0.001

### Set the random seed for reproducible results
torch.manual_seed(0)

# model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = Encoder()
decoder = Decoder()
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)

def add_noise(tensor, poisson_rate=1.0, gaussian_std_dev=0.5):
    gaussian_noise = gaussian_std_dev * torch.randn(tensor.size())
    poisson_noise = torch.poisson(torch.full(tensor.size(), poisson_rate))
    noisy_tensor = tensor + gaussian_noise + poisson_noise
    noisy = torch.clip(noisy_tensor, 0., 1.)
    return noisy



### Training function
def train_epoch_den(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader:  # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        poisson_rate = random.uniform(0.5, 1.5)
        gaussian_std_dev = random.uniform(0.1, 0.9)

        image_noisy = add_noise(image_batch, poisson_rate, gaussian_std_dev)
        image_batch = image_batch.to(device)
        image_noisy = image_noisy.to(device)
        # Encode data
        encoded_data = encoder(image_noisy)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


### Testing function
def test_epoch_den(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:

            poisson_rate = random.uniform(0.5, 1.5)
            gaussian_std_dev = random.uniform(0.1, 0.9)
            # Move tensor to the proper device
            image_noisy = add_noise(image_batch, poisson_rate, gaussian_std_dev)

            image_noisy = image_noisy.to(device)
            # Encode data
            encoded_data = encoder(image_noisy)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


### Training cycle
noise_factor = 0.3
num_epochs = 30
history_da = {'train_loss': [], 'val_loss': []}

for epoch in range(num_epochs):
    print('EPOCH %d/%d' % (epoch + 1, num_epochs))

    ### Training (use the training function)
    train_loss = train_epoch_den(
        encoder=encoder,
        decoder=decoder,
        device=device,
        dataloader=train_loader,
        loss_fn=loss_fn,
        optimizer=optim)
    ### Validation (use the testing function)
    val_loss = test_epoch_den(
        encoder=encoder,
        decoder=decoder,
        device=device,
        dataloader=valid_loader,
        loss_fn=loss_fn)

    # Write train_loss and val_loss to TensorBoard
    writer.add_scalar('Train Loss', train_loss, epoch)
    writer.add_scalar('Validation Loss', val_loss, epoch)
    # Print Validationloss
    history_da['train_loss'].append(train_loss)
    history_da['val_loss'].append(val_loss)
    print('\n EPOCH {}/{} \t train loss {:.5f} \t val loss {:.5f}'.format(epoch + 1, num_epochs, train_loss, val_loss))
# plot_ae_outputs_den(encoder,decoder,noise_factor=noise_factor)
