import cv2
import numpy as np

# Load ground truth and predicted images (should be the same size)
GT = cv2.imread('test_batch0_labels.png')
Pred = cv2.imread('test_batch0_pred.png')

# Ensure that GT and Pred have the same dimensions
if GT.shape != Pred.shape:
    raise ValueError("Ground truth and predicted images must have the same dimensions")

# Initialize variables to store counts
num_classes = np.max(GT) + 1  # Assuming class labels start from 0
correct_per_class = np.zeros(num_classes)
incorrect_per_class = np.zeros(num_classes)
missed_per_class = np.zeros(num_classes)

# Iterate over each class
for class_label in range(num_classes):
    # Create binary masks for the current class
    GT_mask = (GT == class_label)
    Pred_mask = (Pred == class_label)

    # Calculate the number of correctly detected pixels
    correct_pixels = np.sum(np.logical_and(GT_mask, Pred_mask))
    correct_per_class[class_label] = correct_pixels

    # Calculate the number of incorrectly detected pixels
    incorrect_pixels = np.sum(np.logical_and(GT_mask, np.logical_not(Pred_mask)))
    incorrect_per_class[class_label] = incorrect_pixels

    # Calculate the number of missed pixels
    missed_pixels = np.sum(np.logical_and(np.logical_not(GT_mask), Pred_mask))
    missed_per_class[class_label] = missed_pixels

# Print the results
for class_label in range(num_classes):
    print(f"Class {class_label}:")
    print(f"Correct detections: {correct_per_class[class_label]}")
    print(f"Incorrect detections: {incorrect_per_class[class_label]}")
    print(f"Missed detections: {missed_per_class[class_label]}")
    print("\n")

# Optionally, calculate overall metrics
total_correct = np.sum(correct_per_class)
total_incorrect = np.sum(incorrect_per_class)
total_missed = np.sum(missed_per_class)

print("Overall Metrics:")
print(f"Total correct detections: {total_correct}")
print(f"Total incorrect detections: {total_incorrect}")
print(f"Total missed detections: {total_missed}")
