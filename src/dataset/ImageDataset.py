import os
from typing import Any
import gdown
import zipfile
import json
from torch.utils.data import Dataset
import cv2
import numpy as np

class ImageDataset(Dataset):
    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def __init__(self, 
                 dataset_name: str = "train",
                 high_res:str = "1920x1080", 
                 upscale_factors: list = [2], 
                 transforms = None, 
                 download:bool = False,
                 verbose:bool = True):
        super().__init__()
        
        self.dataset_name = dataset_name
        self.upscale_factors = upscale_factors
        self.resources_folder: str = "resources"

        self.high_res_name = high_res

        temp = self.high_res_name.split("x")
        self.high_res_size = (int(temp[0]), int(temp[1]))

        # Compute the folder path and the high resolution path
        self.dir_path = os.path.join(self.resources_folder, self.dataset_name)
        self.high_res_path = os.path.join(self.dir_path, self.high_res_name)

        # Compute the low resolution size, names and paths
        self.low_res_sizes = []
        self.low_res_names = []
        self.low_res_paths = []
        for factor in self.upscale_factors:
            self.low_res_sizes.append((int(temp[0]) // factor, int(temp[1]) // factor))
            self.low_res_names.append(f"{self.low_res_sizes[-1][0]}x{self.low_res_sizes[-1][1]}")
            self.low_res_paths.append(os.path.join(self.dir_path, self.low_res_names[-1]))

        self.transforms = transforms
        self.dataset_link = self.get_link()

        self.chosen_indices = None
        
        self.verbose = verbose
        
        if self.verbose:
            self.print("Check for download and resize ...")

        # Check if the dataset is already downloaded
        if download and not os.path.exists(self.high_res_path):
            if self.verbose:
                print("High-res dataset not present, downloading it ...")
            self.download_dataset(self.dataset_link)

        # Check if the low resolution datasets is already resized
        for i, low_res_path in enumerate(self.low_res_paths):
            if not os.path.exists(low_res_path):
                if self.verbose:
                    print(f"Low-res {self.low_res_names[i]} dataset not present, resizing it ...")

                self.resize_dataset(self.high_res_path, i)
                    
        self.images = os.listdir(self.high_res_path)
    
    def name(self):
        return self.dataset_name

    def number_upscale(self):
        return len(self.upscale_factors)

    def upscale_factor_to_index(self, upscale_factor):
        return self.upscale_factors.index(upscale_factor)
    
    def get_upscale_factor(self, index):
        return self.upscale_factors[index]

    def get_link(self):
        with open("links.json") as f:
            data = json.load(f)
            try:
                return data["datasets"][self.dataset_name][self.high_res_name]
            except:
                raise KeyError(f"{self.high_res_name} dataset link not found")
            
    def open_image(self, path:str):
        return cv2.imread(path)

    def unzip_file(self, file_path: str, extract_path: str):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    def download_dataset(self, url: str):
        folder_path = os.path.join(self.resources_folder, self.dataset_name, self.high_res_name)

        print(folder_path)
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
        new_res_size = self.low_res_sizes[index_upscale]
        new_res_name = self.low_res_names[index_upscale]

        dir_dest = os.path.join(self.dir_path, new_res_name)

        if not os.path.exists(dir_dest):
            os.makedirs(dir_dest)

        images = os.listdir(source)

        self.print("Resizing images...")
        for img in images:
            source_img = os.path.join(source, img)
            dest_img = os.path.join(dir_dest, img)
            try:
                self.resize_image(source_img, dest_img, new_res_size)
            except:
                print(f"Broken Image ({img}), skipping it...")
                os.remove(source_img)
        self.print("Done!")

    def resize_image(self, source:str, dest:str, new_res_size) -> None:
        img = self.open_image(source)
        img = cv2.resize(img, new_res_size)
        cv2.imwrite(dest, img)

    def _move_images(self, images:list, source:str, dest:str) -> None:
        for img in images:
            source_img = os.path.join(source, img)
            dest_img = os.path.join(dest, img)
            os.rename(source_img, dest_img)

    def limit_dataset(self, limit:int) -> None:
        if limit > len(self):
            raise ValueError(f"Limit must be less than {len(self)}")
        
        self.chosen_indices = np.random.choice(len(self), limit, replace=False)

    def reset_dataset_limit(self) -> None:
        self.chosen_indices = None

    def _remove_folder(self, folder_path: str) -> None:
        try:
            os.rmdir(folder_path)
            self.print(f"Folder '{folder_path}' removed successfully.")
        except OSError as e:
            self.print(f"Error: {e}")

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

        high_res = self.open_image(os.path.join(self.high_res_path, self.images[index]))
        if self.transforms is not None:
            high_res = self.transforms(high_res)

        low_res_images = []

        for i, upscale in enumerate(self.upscale_factors):
            low_res = self.open_image(os.path.join(self.low_res_paths[i], self.images[index]))

            if self.transforms is not None:
                low_res = self.transforms(low_res)

            low_res_images.append(low_res)
        
        return low_res_images, high_res

    @staticmethod
    def clean_dir(dir:str = "resources"):
        for d in os.listdir(dir):
            if os.path.isdir(d):
                os.rmdir(d)
    
    def print_info(self):
        # Display the sizes of the dataset
        dir_train_high = os.path.join(self.resources_folder, self.dataset_name, self.high_res_name)

        self.print(f'Number of train high ({self.high_res_name}) resolution images: {len(os.listdir(dir_train_high))}')
        for i, low_res_path in enumerate(self.low_res_paths):
            self.print(f'Number of train low ({self.low_res_names[i]}) resolution images: {len(os.listdir(low_res_path))}')

if __name__ == "__main__":

    
    import matplotlib.pyplot as plt
    import torchvision
    import numpy as np
    
    # CarlaDataset.clean_dir("resources")
    test = ImageDataset(dataset_name="train", 
                        high_res="1920x1080",  
                        upscale_factors=[2, 4, 8],
                        transforms = torchvision.transforms.ToTensor(), download=False)
    
    test.print_info()
    print(len(test))
    print(test[0][0].shape)
    print(test[0][1].shape)
    print(test[1][0].shape)
    print(test[1][1].shape)
    print(test[-1][0].shape)
    print(test[-1][1].shape)
    
    test.print_info()
    low_res, high_res, upscale_factor = test[0]
    


    plt.imshow(torchUtil.tensor_to_image(low_res))
    plt.show()
    plt.imshow(torchUtil.tensor_to_image(high_res))
    plt.show()