from src.utils.PytorchUtil import PytorchUtil as torchUtil

from torch.utils.data import Dataset
from typing import Any

import os
import gdown
import zipfile
import json
import cv2
import numpy as np
import torch

class ImageDataset(Dataset):
    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def __init__(self, 
                 dataset_name: str = "train",
                 hr_name:str = "1920x1080", 
                 upscale_factors: list = [2], 
                 channels: list = ["r", "g", "b"],
                 transforms = None, 
                 download:bool = False,
                 verbose:bool = True,
                 seed:int = None):
        super().__init__()
        
        self.dataset_name: str = dataset_name
        self.upscale_factors: list = upscale_factors

        self.channels: list = channels
        self.channels_used: dict = None

        self.resources_folder: str = "resources"

        self.hr_name: str = hr_name

        temp = self.hr_name.split("x")
        self.hr_data_size: tuple = (int(temp[0]), int(temp[1]))

        # Compute the folder path and the high resolution path
        self.dir_path = os.path.join(self.resources_folder, self.dataset_name)
        self.hr_path = os.path.join(self.dir_path, self.hr_name)

        # Compute the low resolution size, names and paths
        self.lr_sizes = []
        self.lr_names = []
        self.lr_paths = []

        for factor in self.upscale_factors:
            self.lr_sizes.append((int(temp[0]) // factor, int(temp[1]) // factor))
            self.lr_names.append(f"{self.lr_sizes[-1][0]}x{self.lr_sizes[-1][1]}")
            self.lr_paths.append(os.path.join(self.dir_path, self.lr_names[-1]))

        self.transforms = transforms
        self.dataset_link = None
        self.load_dataset_info()

        self.chosen_indices = None
        
        self.verbose = verbose
        
        if self.verbose:
            self.print("Check for download and resize ...")

        # Check if the dataset is already downloaded
        if download and not os.path.exists(self.hr_path):
            if self.verbose:
                print("High-res dataset not present, downloading it ...")
            self.download_dataset(self.dataset_link)

        # Check if the low resolution datasets is already resized
        for i, lr_patch in enumerate(self.lr_paths):
            if not os.path.exists(lr_patch):
                if self.verbose:
                    print(f"Low-res {self.lr_names[i]} dataset not present, resizing it ...")

                self.resize_dataset(self.hr_path, i)
                    
        self.images = os.listdir(self.hr_path)

    def name(self):
        return self.dataset_name

    def number_upscale(self):
        return len(self.upscale_factors)

    def upscale_factor_to_index(self, upscale_factor):
        return self.upscale_factors.index(upscale_factor)
    
    def get_upscale_factor(self, index):
        return self.upscale_factors[index]

    def load_dataset_info(self):
        with open("links.json") as f:
            data = json.load(f)
            try:
                dataset = data["datasets"][self.dataset_name]
                self.dataset_link = dataset[self.hr_name]
                self.channels_position = dataset["channels"]
            except:
                raise KeyError(f"{self.hr_name} dataset link not found")
            
    def load_data_from_path(self, path:str):
        return torchUtil.open_data(path)
        
    # Take only the interesting channels
    def filter_channels_to_image(self, data):
        return torchUtil.filter_data(data, self.channels_used, self.channels_position)
        
    def save_image(self, path:str, image):
        if path.endswith(".png"):
            cv2.imwrite(path, image)
        elif path.endswith(".npy"):
            np.save(path, image)

    def unzip_file(self, file_path: str, extract_path: str):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    def download_dataset(self, url: str):
        folder_path = os.path.join(self.resources_folder, self.dataset_name, self.hr_name)

        zip_path = folder_path + ".zip"

        if os.path.exists(folder_path):
            self.print(f"Dataset already downloaded at {folder_path}")
            return
        else:
            os.makedirs(folder_path)
        
        gdown.download(url, zip_path, quiet=False)

        self.print(f"Extracting to {folder_path} ...")
        self.unzip_file(zip_path, folder_path)
        
        os.remove(zip_path)

        # Get the folder name existing after unzip
        folder_name = os.listdir(folder_path)[0]
        # Get all the images and cut paste them in the folder_path
        images = os.listdir(os.path.join(folder_path, folder_name))
        self._move_images(images, os.path.join(folder_path, folder_name), folder_path)

        # Remove the folder_name folder
        self._remove_folder(os.path.join(folder_path, folder_name))
        
        self.print("Done!")

    def resize_dataset(self, source:str, index_upscale: int) -> None:
        new_res_size = self.lr_sizes[index_upscale]
        new_res_name = self.lr_names[index_upscale]

        dir_dest = os.path.join(self.dir_path, new_res_name)

        if not os.path.exists(dir_dest):
            os.makedirs(dir_dest)

        images = os.listdir(source)

        self.print("Resizing images...")
        for img in images:
            source_img = os.path.join(source, img)
            dest_img = os.path.join(dir_dest, img)
            self.resize_image(source_img, dest_img, new_res_size)

            try:
                pass
            except:
                print(f"Broken Image ({img}), skipping it...")
                os.remove(source_img)

        self.print("Done!")

    def resize_image(self, source:str, dest:str, new_res_size) -> None:
        img = self.load_data_from_path(source)

        result = torchUtil.resize_data(img, new_res_size, self.channels, self.channels_position)
            
        self.save_image(dest, result)

    def _move_images(self, images:list, source:str, dest:str) -> None:
        for img in images:
            source_img = os.path.join(source, img)
            dest_img = os.path.join(dest, img)
            os.rename(source_img, dest_img)

    def _remove_folder(self, folder_path: str) -> None:
        try:
            os.rmdir(folder_path)
            self.print(f"Folder '{folder_path}' removed successfully.")
        except OSError as e:
            self.print(f"Error: {e}")

    def limit_dataset(self, limit:int) -> None:
        if limit > len(self):
            raise ValueError(f"Limit must be less than {len(self)}")
        
        self.chosen_indices = np.random.choice(len(self), limit, replace=False)

    def reset_dataset_limit(self) -> None:
        self.chosen_indices = None

    def check_index(self, index):
        if index < 0:
            index = len(self) + index

        if self.chosen_indices is not None:
            index = self.chosen_indices[index]

        return index

    def __len__(self):
        return len(self.images) if self.chosen_indices is None else len(self.chosen_indices)

    """
    Indices are as follow, 0 -> high_res[0] + low_res[0][0] + upscale_factor[0], 1 -> high_res[0] + low_res[0][1] + upscale_factor[1]
    """
    def __getitem__(self, index) -> Any:
        index = self.check_index(index)

        hr_data_np = self.load_data_from_path(os.path.join(self.hr_path, self.images[index]))
        
        if self.transforms is None:
            hr_data_tensor = torchUtil.numpy_to_tensor(hr_data_np)
        else:
            hr_data_tensor = self.transforms(hr_data_np)
            
        hr_img_tensor = self.filter_channels_to_image(hr_data_tensor)

        lr_data_tensors = []

        for i, upscale in enumerate(self.upscale_factors):
            lr_data_np = self.load_data_from_path(os.path.join(self.lr_paths[i], self.images[index]))

            if self.transforms is None:
                lr_data_tensor = torchUtil.numpy_to_tensor(lr_data_np)
            else:
                lr_data_tensor = self.transforms(lr_data_np)

            lr_data_tensors.append(lr_data_tensor)

        return lr_data_tensors, hr_img_tensor

    @staticmethod
    def clean_dir(dir:str = "resources"):
        for d in os.listdir(dir):
            if os.path.isdir(d):
                os.rmdir(d)
    
    def print_info(self):
        # Display the sizes of the dataset
        dir_train_high = os.path.join(self.resources_folder, self.dataset_name, self.hr_name)

        self.print(f'Number of train high ({self.hr_name}) resolution images: {len(os.listdir(dir_train_high))}')
        for i, low_res_path in enumerate(self.lr_paths):
            self.print(f'Number of train low ({self.lr_names[i]}) resolution images: {len(os.listdir(low_res_path))}')

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    
    path = "resources/train_full_channel/1920x1080/0.npy"

    img = np.load(path)
    img = img[:, :, :3]

    x = 8

    # resize and devide by x each channel
    dest_img = np.zeros((img.shape[0] // x, img.shape[1] // x, img.shape[2]), dtype=np.uint8)

    for i in range(img.shape[2]):
        dest_img[:, :, i] = cv2.resize(img[:, :, i], (img.shape[1] // x, img.shape[0] // x), interpolation=cv2.INTER_CUBIC)
    
    plt.imshow(dest_img)
    plt.show()

    