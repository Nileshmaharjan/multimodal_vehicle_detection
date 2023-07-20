import os
import shutil


def list_and_copy_images(source_folder, destination_folder):
    try:
        os.makedirs(destination_folder, exist_ok=True)

        # List all files in the source folder
        files = os.listdir(source_folder)

        for file in files:
            if file.endswith(".png") and "_ir.png" in file:
                source_path = os.path.join(source_folder, file)
                destination_path = os.path.join(destination_folder, file)
                shutil.copy(source_path, destination_path)
                print(f"Successfully copied {file} to the destination folder.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Replace these paths with your source and destination folders
    source_folder = r'C:/Users/User/Documents/Projects/Nilesh/uav/SuperYOLO/dataset/original/VEDAI_1024/images/class_1/'
    destination_folder =  r"C:/Users/User/Documents/Projects/Nilesh/fso_traffic_surveillance/autoencoder/images/ir/"

    list_and_copy_images(source_folder, destination_folder)


mean [0.43069887 0.48378574 0.48171843]
std [0.16868428 0.16128027 0.17712247]

mean [0.66828259 0.66828259 0.66828259]
std [0.14035617 0.14035617 0.14035617]