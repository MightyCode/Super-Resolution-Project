import os
import sys
import gdown
import zipfile
from sklearn.model_selection import train_test_split

resources_folder: str = "resources"
carla_folder: str = os.path.join(resources_folder, "carla")  # Create 'carla' folder within 'resources'

def down_drive(url: str, dest: str):
    gdown.download(url, dest, quiet=False)

def unzip_file(file_path: str, extract_path: str):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def get_dataset(url: str, dest: str):
    folder_path = os.path.join(carla_folder, dest)  # Save datasets directly into 'carla' folder
    zip_path = folder_path + ".zip"
    if os.path.exists(folder_path):
        print(f"Dataset already downloaded at {folder_path}")
        return
    down_drive(url=url, dest=zip_path)
    print(f"Extracting to {folder_path} ...")
    unzip_file(zip_path, carla_folder)
    os.remove(zip_path)
    print("Done!")

def remove_folder(folder_path):
    try:
        os.rmdir(folder_path)
        print(f"Folder '{folder_path}' removed successfully.")
    except OSError as e:
        print(f"Error: {e}")


def organize_dataset(split: float):
    os.makedirs(carla_folder, exist_ok=True)  # Create 'carla' folder if it doesn't exist

    high_res_url = "https://drive.google.com/uc?export=download&id=16cXZ2etHDMrgXSXtLlZ5HuPh7e0KSkVk"
    low_res_url = "https://drive.google.com/uc?export=download&id=19WNetKoHChsVk4KDBAgnhLQEcS2ILgId"

    high_res_dest = "256.zip"
    low_res_dest = "128.zip"

    get_dataset(high_res_url, high_res_dest)
    get_dataset(low_res_url, low_res_dest)

    split_dataset(split)

    remove_folder(os.path.join(carla_folder, os.path.splitext(high_res_dest)[0]))
    remove_folder(os.path.join(carla_folder, os.path.splitext(low_res_dest)[0]))

def split_dataset(split: float):
    carla_train_folder = os.path.join(carla_folder, "train")
    carla_valid_folder = os.path.join(carla_folder, "valid")

    os.makedirs(carla_train_folder, exist_ok=True)
    os.makedirs(carla_valid_folder, exist_ok=True)

    high_res_folder = os.path.join(carla_folder, "256")
    low_res_folder = os.path.join(carla_folder, "128")

    high_res_images = os.listdir(high_res_folder)
    low_res_images = os.listdir(low_res_folder)

    high_res_train, high_res_valid = train_test_split(high_res_images, train_size=split, random_state=42)
    low_res_train, low_res_valid = train_test_split(low_res_images, train_size=split, random_state=42)

    organize_images(high_res_folder, high_res_train, carla_train_folder, "high_res")
    organize_images(low_res_folder, low_res_train, carla_train_folder, "low_res")

    organize_images(high_res_folder, high_res_valid, carla_valid_folder, "high_res")
    organize_images(low_res_folder, low_res_valid, carla_valid_folder, "low_res")

def organize_images(source_folder: str, image_list: list, destination_folder: str, subfolder: str):
    subfolder_path = os.path.join(destination_folder, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)

    for image in image_list:
        source_path = os.path.join(source_folder, image)
        destination_path = os.path.join(subfolder_path, image)
        os.rename(source_path, destination_path)

if __name__ == "__main__":
    split = 0.8 if len(sys.argv) < 2 else float(sys.argv[1])
    organize_dataset(split)
