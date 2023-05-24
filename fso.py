

import cv2
import numpy as np

# Load the two images
img1 = cv2.imread('uav.jpg')
img2 = cv2.imread('lognormal.jpg')

# Convert the images to grayscale
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Calculate the mean and standard deviation of the pixel values in both images
mean_img1, std_dev_img1 = cv2.meanStdDev(gray_img1)
mean_img2, std_dev_img2 = cv2.meanStdDev(gray_img2)

# Calculate the Signal-to-Noise Ratio (SNR) of both images
snr_img1 = 20 * np.log10(mean_img1 / std_dev_img1)
snr_img2 = 20 * np.log10(mean_img2 / std_dev_img2)

# Compare the SNR values of both images
if snr_img1 == snr_img2:
    print("The noise levels of the two images are the same.")
    print("SNR: ", snr_img1)
else:
    print("The noise levels of the two images are different.")
    print("SNR of orignal: ", snr_img1)
    print("SNR of noisy_uav: ", snr_img2)