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

combined_data_path = [data_path_1, data_path_2, data_path_3]

train_transform3 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# Combine datasets
combined_datasets = []
for path in combined_data_path:
    dataset = torchvision.datasets.ImageFolder(root=path, transform=train_transform3)
    combined_datasets.append(dataset)
combined_dataset = ConcatDataset(combined_datasets)

# Define the length of the combined dataset
combined_m = len(combined_dataset)
print(combined_m)
# Split the combined dataset with 70% training and 30% validation
combined_train_data, combined_val_data = random_split(combined_dataset, [84576, 36247])







batch_size = 32

# Use DataLoader for the combined datasets
train_loader = DataLoader(combined_train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(combined_val_data, batch_size=batch_size)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

import torch
import torch.nn as nn
import torch.nn.functional as F



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


# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('Cuda available:', torch.cuda.is_available())
print(f'Selected device: {device}')

# Instantiate the Autoencoder
autoencoder = Autoencoder()

# Define the loss function
loss_fn = torch.nn.MSELoss()

# Define an optimizer
lr = 0.001
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=1e-07)



# Move the autoencoder to the selected device
autoencoder.to(device)

# Function to save model checkpoint

checkpoint_path = "C:/Users/User/Documents/Projects/Nilesh/fso_traffic_surveillance/autoencoder/checkpoint/model_checkpoint_epoch_23.pt"
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
        poisson_rate = random.uniform(0.01, 1.00)
        gaussian_std_dev = random.uniform(0.01, 1.00)


        image_noisy = add_noise(image_batch, poisson_rate, gaussian_std_dev)
        image_batch = image_batch.to(device)
        image_noisy = image_noisy.to(device)
        # Encode and Decode data
        decoded_data = autoencoder(image_noisy)
        decoded_data = decoded_data.to(device)


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
            poisson_rate = random.uniform(0.01, 1.00)
            gaussian_std_dev = random.uniform(0.01, 1.00)

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

