import numpy as np

# Load the NumPy array from the saved file (replace 'my_array.npy' with your file name)
file_name = "../edsr_confusion_matrix_high.npy"
loaded_arr = np.load(file_name)



# Calculate accuracy for each class
class_accuracies = []
for i in range(len(loaded_arr)):
    true_positives = loaded_arr[i, i]
    total_samples = np.sum(loaded_arr[i])
    accuracy_i = true_positives / total_samples
    class_accuracies.append(accuracy_i)

# Calculate overall accuracy
overall_accuracy = np.sum(np.diag(loaded_arr)) / np.sum(loaded_arr)

print("Accuracy for each class:")
for i, accuracy_i in enumerate(class_accuracies):
    print(f"Class {i+1}: {accuracy_i:.2f}")

print(f"Overall Accuracy: {overall_accuracy:.2f}")