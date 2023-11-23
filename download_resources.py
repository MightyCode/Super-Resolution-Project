import os
import sys
import gdown
import zipfile
from sklearn.model_selection import train_test_split
import json
from torch.utils.data import Dataset
from cv2 import imread, imwrite, resize

class CarlaDataset(Dataset):
    def __init__(self, high_res:str = "1920x1080", low_res:str = "1280x720", split:str = "train", download:bool = False):
        super().__init__()
        self.split = split
        self.resources_folder: str = "resources"
        self.high_res = high_res
        self.low_res = low_res
        with open("links.json") as f:
            data = json.load(f)
            try:
                self.dataset_link = data["datasets"][high_res]
            except:
                raise KeyError(f"{high_res} dataset link not found")

        if not download and not self.check_download():
            raise FileNotFoundError("You didn't download the dataset, you can do it by specifying download=True")
        else:
            self.download_dataset(self.dataset_link, self.high_res)
            self.resize_dataset()
            # self.split_dataset()

    def check_download(self) -> bool:
        return os.path.exists(os.path.join(self.resources_folder, self.high_res))

    def down_drive(self, url: str, dest: str):
        gdown.download(url, dest, quiet=False)

    def unzip_file(self, file_path: str, extract_path: str):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    def download_dataset(self, url: str, dest: str):
        folder_path = os.path.join(self.resources_folder, dest)
        zip_path = folder_path + ".zip"
        if os.path.exists(folder_path):
            print(f"Dataset already downloaded at {folder_path}")
            return
        gdown.download(url, zip_path, quiet=False)
        print(f"Extracting to {folder_path} ...")
        self.unzip_file(zip_path, folder_path)
        os.remove(zip_path)
        print("Done!")

    def resize_dataset(self, source:str, dest:str):
        source_folder = os.path.join(self.resources_folder, source)
        dest_folder = os.path.join(self.resources_folder, dest)
        destx, desty = dest.split("x")
        print("Resizing images ...")
        for img in os.listdir(source_folder):
            source_img = imread(os.path.join(source_folder, img))
            dest_img = resize(source_img,(destx, desty))
            imwrite(os.path.join(dest_folder, img), dest_img)
        print("Done!")


    def remove_folder(self, folder_path: str):
        try:
            os.rmdir(folder_path)
            print(f"Folder '{folder_path}' removed successfully.")
        except OSError as e:
            print(f"Error: {e}")

    def organize_dataset(self, split: float=0.8):
        os.makedirs(self.carla_folder, exist_ok=True)  # Create 'carla' folder if it doesn't exist

        high_res_url = "https://drive.google.com/uc?export=download&id=16cXZ2etHDMrgXSXtLlZ5HuPh7e0KSkVk"
        low_res_url = "https://drive.google.com/uc?export=download&id=19WNetKoHChsVk4KDBAgnhLQEcS2ILgId"

        high_res_dest = "256.zip"
        low_res_dest = "128.zip"

        self.get_dataset(high_res_url, high_res_dest)
        self.get_dataset(low_res_url, low_res_dest)

        self.split_dataset(split)

        self.remove_folder(os.path.join(self.carla_folder, os.path.splitext(high_res_dest)[0]))
        self.remove_folder(os.path.join(self.carla_folder, os.path.splitext(low_res_dest)[0]))

    def split_dataset(self, split: float):
        carla_train_folder = os.path.join(self.carla_folder, "train")
        carla_valid_folder = os.path.join(self.carla_folder, "valid")

        os.makedirs(carla_train_folder, exist_ok=True)
        os.makedirs(carla_valid_folder, exist_ok=True)

        high_res_folder = os.path.join(self.carla_folder, "256")
        low_res_folder = os.path.join(self.carla_folder, "128")

        high_res_images = os.listdir(high_res_folder)
        low_res_images = os.listdir(low_res_folder)

        high_res_train, high_res_valid = train_test_split(high_res_images, train_size=split, random_state=42)
        low_res_train, low_res_valid = train_test_split(low_res_images, train_size=split, random_state=42)

        self.organize_images(high_res_folder, high_res_train, carla_train_folder, "high_res")
        self.organize_images(low_res_folder, low_res_train, carla_train_folder, "low_res")

        self.organize_images(high_res_folder, high_res_valid, carla_valid_folder, "high_res")
        self.organize_images(low_res_folder, low_res_valid, carla_valid_folder, "low_res")

    def organize_images(self, source_folder: str, image_list: list, destination_folder: str, subfolder: str):
        subfolder_path = os.path.join(destination_folder, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)

        for image in image_list:
            source_path = os.path.join(source_folder, image)
            destination_path = os.path.join(subfolder_path, image)
            os.rename(source_path, destination_path)

if __name__ == "__main__":
    # split = 0.8 if len(sys.argv) < 2 else float(sys.argv[1])
    test = CarlaDataset("256x256", "128x128", download=True)
    # test.organize_dataset(split)