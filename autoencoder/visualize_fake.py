# import torch
# import torchvision
# import matplotlib.pyplot as plt
# from torchvision import transforms
#
# train_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# train_dataset = torchvision.datasets.CIFAR10(root='D:/Research/Super/autoencoder/dataset/', download=True, transform=train_transform)
#
#
# # Set the number of images to visualize
# num_images = 5
#
# # Create a figure with subplots
# fig, axes = plt.subplots(1, num_images, figsize=(12, 4))
#
# # Iterate over the dataset and plot sample images
# for i in range(num_images):
#     image, _ = train_dataset[i]  # Get the i-th image and ignore the label
#     image = image.permute(1, 2, 0)  # Reorder dimensions from (C, H, W) to (H, W, C)
#     image = image.numpy()  # Convert to NumPy array
#     image = image.clip(0, 1)  # Clip values between 0 and 1
#
#     axes[i].imshow(image)
#     axes[i].axis('off')
#
# plt.tight_layout()
# plt.show()