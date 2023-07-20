import numpy as np  # this module is useful to work with numerical arrays
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import random
from tqdm import tqdm  # Import tqdm for the progress bar

# Set the directory for TensorBoard logs
log_dir = r"C:/Users/User/Documents/Projects/Nilesh/fso_traffic_surveillance/autoencoder/runs/"


# Create a SummaryWriter object
writer = SummaryWriter(log_dir=log_dir)

data_path_1 = r"C:/Users/User/Documents/Projects/Nilesh/fso_traffic_surveillance/autoencoder/images/color/"
data_path_2 = r"C:/Users/User/Documents/Projects/Nilesh/fso_traffic_surveillance/autoencoder/images/ir/"


train_transform1 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_transform2 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset1 = torchvision.datasets.ImageFolder(root=data_path_1, transform=train_transform1)
train_dataset2 = torchvision.datasets.ImageFolder(root=data_path_2, transform=train_transform2)

# Define the length of each dataset
m1 = len(train_dataset1)
m2 = len(train_dataset2)

# Split the datasets
train_data1, val_data1 = random_split(train_dataset1, [int(0.75*m1), int(0.25*m1)])
train_data2, val_data2 = random_split(train_dataset2, [int(0.75*m2), int(0.25*m2)])

# Combine the train datasets
combined_train_data = ConcatDataset([train_data1, train_data2])
combined_val_data = ConcatDataset([val_data1, val_data2])

batch_size = 32

train_loader = torch.utils.data.DataLoader(combined_train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(combined_val_data, batch_size=batch_size)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            # nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode data
        x = self.encoder(x)
        # Decode data
        x = self.decoder(x)
        return x

# Instantiate the Autoencoder
autoencoder = Autoencoder()

# Define the loss function
loss_fn = torch.nn.MSELoss()

# Define an optimizer
lr = 0.0001
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=1e-03)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('Cuda available:', torch.cuda.is_available())
print(f'Selected device: {device}')

# Move the autoencoder to the selected device
autoencoder.to(device)

# Function to save model checkpoint
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


def psnr(original, denoised):
    mse = torch.mean((original - denoised) ** 2)
    max_pixel = 1.0  # Assuming pixel values are in the range [0, 1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr


def train_epoch_den(autoencoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for the autoencoder
    autoencoder.train()
    train_loss = []
    psnr_list = []
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
        decoded_data = decoded_data.to(device)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)

        # Calculate PSNR
        psnr_batch = psnr(image_batch, decoded_data)
        psnr_list.append(psnr_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.item())

    mean_psnr = torch.mean(torch.stack(psnr_list))

    return np.mean(train_loss), mean_psnr.item()

def test_epoch_den(autoencoder, device, dataloader, loss_fn):
    # Set evaluation mode for the autoencoder
    autoencoder.eval()
    val_loss = []
    psnr_list = []
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
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

            # Append the network output and the original image to the lists
            conc_out.append(decoded_data)
            conc_label.append(image_batch)

            psnr_batch = psnr(image_batch, decoded_data)
            psnr_list.append(psnr_batch)

        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)

        mean_psnr = torch.mean(torch.stack(psnr_list))
    return val_loss.data,  mean_psnr.item()



### Training cycle
noise_factor = 0.3
num_epochs = 200
history_da = {'train_loss': [], 'val_loss': [], 'train_psnr': [], 'val_psnr': []}


for epoch in range(num_epochs):
    print('EPOCH %d/%d' % (epoch + 1, num_epochs))

    # Initialize tqdm for the training data loader
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)")

    ### Training (use the training function)
    train_loss, train_psnr  = train_epoch_den(
        autoencoder=autoencoder,
        device=device,
        dataloader=train_loader_tqdm,
        loss_fn=loss_fn,
        optimizer=optimizer)

    # Close the tqdm progress bar for the training data loader
    train_loader_tqdm.close()

    # Initialize tqdm for the validation data loader
    valid_loader_tqdm = tqdm(valid_loader, desc=f"Epoch {epoch + 1} (Validation)")

    ### Validation (use the testing function)
    val_loss, val_psnr = test_epoch_den(
        autoencoder=autoencoder,
        device=device,
        dataloader=valid_loader_tqdm,
        loss_fn=loss_fn)

    # Close the tqdm progress bar for the validation data loader
    valid_loader_tqdm.close()

    # Write train_loss and val_loss to TensorBoard
    writer.add_scalar('Train Loss', train_loss, epoch)
    writer.add_scalar('Validation Loss', val_loss, epoch)
    writer.add_scalar('Train PSNR', train_psnr, epoch)
    writer.add_scalar('Validation PSNR', val_psnr, epoch)


    save_checkpoint(epoch, autoencoder, optimizer, val_loss)

    # Print Validation loss
    history_da['train_loss'].append(train_loss)
    history_da['val_loss'].append(val_loss)
    history_da['train_psnr'].append(train_psnr)
    history_da['val_psnr'].append(val_psnr)

    print('\n EPOCH {}/{} \t train loss {:.5f} \t val loss {:.5f} \t train PSNR {:.2f} \t val PSNR {:.2f}'.format(
        epoch + 1, num_epochs, train_loss, val_loss, train_psnr, val_psnr))
