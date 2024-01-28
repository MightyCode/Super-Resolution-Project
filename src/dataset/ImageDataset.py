from src.utils.PytorchUtil import PytorchUtil as torchUtil

from torch.utils.data import Dataset
from typing import Any

import os
import gdown
import zipfile
import json
import cv2
import numpy as np

"""
Dataset class for the image super resolution
Image are data of shape (C, H, W) where C is the number of channels.
For example tensor of shape (3, 1080, 1920) is a RGB image of size 1920x1080
"""
class ImageDataset(Dataset):

    """
    Override the print method in order to print only if verbose is True
    """
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
                 verbose:bool = True):
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

    """
    Get the dataset related information from the links.json file where all the datasets information are stored
    """
    def load_dataset_info(self):
        with open("links.json") as f:
            data = json.load(f)
            try:
                dataset = data["datasets"][self.dataset_name]
                self.dataset_link = dataset[self.hr_name]
                self.reference_channel_positions = dataset["reference_channel_positions"]
                self.channel_positions = dataset["channel_positions"]
                self.channel_downresolution_methods = dataset["channel_downresolution_methods"]
                self.channel_superresolution_methods = dataset["channel_superresolution_methods"]
            except:
                raise KeyError(f"{self.hr_name} dataset link not found")
    
    def get_channel_positions(self, channel: str = None):
        if channel is not None:
            return self.channel_positions[channel]
        
        return self.channel_positions
    

    """
    Return the method used to downsample the channel when creating the low resolution dataset
    """
    def get_channel_downresolution_method(self, channel: str = None):
        if channel is not None:
            return self.channel_downresolution_methods[channel]
        
        return self.channel_downresolution_methods
    

    def get_channel_superresolution_method(self, channel: str = None):
        if channel is not None:
            return self.channel_superresolution_methods[channel]

        return self.channel_superresolution_methods


    def load_data_from_path(self, path:str):
        return torchUtil.open_data(path)
        
    """
    Take only the interesting channels in order to tranform data of shape (C, H, W) to (3, H, W)
    """
    def filter_channels_to_image(self, data, use_channels=False):
        if use_channels:
            return torchUtil.filter_data_to_img(data, self.channel_positions, self.channels)
        
        return torchUtil.filter_data_to_img(data, self.channel_positions)
    
    """
    Save the image in the correct format
    """
    def save_image(self, path:str, image):
        if path.endswith(".png"):
            cv2.imwrite(path, image)
        elif path.endswith(".npy"):
            np.save(path, image)
        else:
            raise ValueError("The image format is not supported")

    """
    Unzip the file at file_path and extract it to extract_path
    """
    def unzip_file(self, file_path: str, extract_path: str):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    """
    Download the dataset from the link and extract it to the resources folder
    """
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

    """
    Resize the dataset at source and save it at the destination
    """
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
            self.resize_data(source_img, dest_img, new_res_size)

            try:
                pass
            except:
                print(f"Broken Image ({img}), skipping it...")
                os.remove(source_img)

        self.print("Done!")

    """
    Resize the data at source and save it at the destination
    """
    def resize_data(self, source: str, dest: str, new_res_size) -> None:
        img = self.load_data_from_path(source)

        result = torchUtil.shrink_data(
            img, new_res_size,
            self.reference_channel_positions, self.channel_positions, self.channel_downresolution_methods)
            
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


    """
    Limit the size of the dataset to a certain number of data.
    Choose randomly the data to keep
    """
    def limit_dataset(self, limit: int) -> None:
        if limit > len(self):
            raise ValueError(f"Limit must be less than {len(self)}")
        
        self.chosen_indices = np.random.choice(len(self), limit, replace=False)

    """
    Remove the limit on the dataset size
    """
    def reset_dataset_limit(self) -> None:
        self.chosen_indices = None

    """
    Check if the index is valid, if it is negative, it will be converted to a positive index
    If dataset is limited, the index will be converted to the chosen_indices index
    """
    def check_index(self, index: int):
        if index < 0:
            index = len(self) + index

        if self.chosen_indices is not None:
            index = self.chosen_indices[index]

        return index

    def __len__(self):
        return len(self.images) if self.chosen_indices is None else len(self.chosen_indices)

    """
    Return the data of all low resolution and the high resolution data
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

            # In case we are modyfing the channels, we need to filter the channels to get the image
            lr_data_tensor = self.filter_channels_to_image(lr_data_tensor, use_channels=True)

            lr_data_tensors.append(lr_data_tensor)

        return lr_data_tensors, hr_img_tensor
    
    """
    Print some useful information about the dataset
    """
    def print_info(self):
        # Display the sizes of the dataset
        dir_train_high = os.path.join(self.resources_folder, self.dataset_name, self.hr_name)

        self.print(f'Number of train high ({self.hr_name}) resolution images: {len(os.listdir(dir_train_high))}')
        for i, low_res_path in enumerate(self.lr_paths):
            self.print(f'Number of train low ({self.lr_names[i]}) resolution images: {len(os.listdir(low_res_path))}')

    @staticmethod
    def clean_dir(dir: str = "resources"):
        for d in os.listdir(dir):
            if os.path.isdir(d):
                os.rmdir(d)

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

    