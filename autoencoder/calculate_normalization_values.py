import os
import cv2
import numpy as np

def compute_mean_std_color_images(images_folder):
    # Load all color images from the folder into a list
    image_list = []
    i = 0;
    for filename in os.listdir(images_folder):
        print(i +1)
        image_path = os.path.join(images_folder, filename)
        image = cv2.imread(image_path)
        image_list.append(image)
        i = i +1


    # Convert the list of images to a NumPy array
    image_array = np.array(image_list)

    # Calculate mean and standard deviation along each color channel (across all images)
    mean = np.mean(image_array, axis=(0, 1, 2)) / 255
    std = np.std(image_array, axis=(0, 1, 2)) / 255

    print('mean', mean)
    print('std', std)

    return mean, std

# Replace 'images_folder' with the path to the folder containing your color images
mean, std = compute_mean_std_color_images(r"C:/Users/User/Documents/Projects/Nilesh/fso_traffic_surveillance/autoencoder/images/ir/")

print(f"Mean pixel values (across RGB channels): {mean}")
print(f"Standard deviation of pixel values (across RGB channels): {std}")
