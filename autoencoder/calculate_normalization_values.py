import os
import cv2
import numpy as np

def compute_mean_std_color_images(images_folder):
    # Initialize variables to store cumulative sums
    num_images = 0
    cumulative_mean = np.zeros(3)
    cumulative_std = np.zeros(3)

    for filename in os.listdir(images_folder):
        # Read the image
        image_path = os.path.join(images_folder, filename)
        image = cv2.imread(image_path)

        # Check if the image was successfully read
        if image is not None:
            # Convert BGR image to RGB
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize the image to the target size
            image_resized = cv2.resize(image, (256, 256))

            # Incrementally update cumulative sum of pixel values
            cumulative_mean += np.mean(image_resized, axis=(0, 1)) / 255
            cumulative_std += np.std(image_resized, axis=(0, 1)) / 255

            num_images += 1

        # if num_images >= 30000:
        #     break

    # Calculate the mean and standard deviation based on cumulative sums
    mean = cumulative_mean / num_images
    std = cumulative_std / num_images

    print('mean', mean)
    print('std', std)

    return mean, std

# Replace 'images_folder' with the path to the folder containing your color images
mean, std = compute_mean_std_color_images(r"C:/Users/User/Documents/Projects/Nilesh/fso_traffic_surveillance/autoencoder/images/coco/class_1/")

print(f"Mean pixel values (across RGB channels): {mean}")
print(f"Standard deviation of pixel values (across RGB channels): {std}")


# RGB
# mean [0.46961793 0.44640928 0.40719114]
# std [0.23942938 0.23447396 0.23768907]

#BGR
# mean [0.40719114 0.44640928 0.46961793]
# std [0.23768907 0.23447396 0.23942938]
