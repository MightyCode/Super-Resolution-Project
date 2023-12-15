import os
from typing import Any
import gdown
import zipfile
import json
from torch.utils.data import Dataset
import cv2
import numpy as np
import math

class CarlaDatasetMultipleSize(Dataset):
    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def __init__(self, 
                 res_1:str = "1920x1080", 
                 res_2:str = "1280x720", 
                 res_3:str = "1280x720", 
                 split:str = "train", 
                 transforms = None, 
                 download:bool = False,
                 verbose:bool = True):
        super().__init__()
        
        self.split = split
        self.resources_folder: str = "resources"

        self.res_1 = res_1
        self.res_2 = res_2
        self.res_3 = res_3

        self.train = "train"
        self.test = "test"

        temp = res_1.split("x")
        self.res_1_size = (int(temp[0]), int(temp[1]))

        temp = res_2.split("x")
        self.res_2_size = (int(temp[0]), int(temp[1]))

        temp = res_3.split("x")
        self.res_3_size = (int(temp[0]), int(temp[1]))

        self.upscale_factor_1 = int(self.res_1.split("x")[0]) / int(self.res_3.split("x")[0])
        self.upscale_factor_2 = int(self.res_1.split("x")[0]) / int(self.res_2.split("x")[0])
        self.upscale_factor_3 = int(self.res_2.split("x")[0]) / int(self.res_3.split("x")[0])

        self.transforms = transforms
        self.dataset_link = self.get_link(res_1)

        self.dir_path = os.path.join(self.resources_folder, self.split)
        self.res_1_path = os.path.join(self.dir_path, self.res_1)
        self.res_2_path = os.path.join(self.dir_path, self.res_2)
        self.res_3_path = os.path.join(self.dir_path, self.res_3)

        self.chosen_indices = None
        
        self.verbose = verbose
        
        if self.verbose:
            self.print("Check for download and resize ...")

        if download and not os.path.exists(self.res_1_path):
            self.download_dataset(self.dataset_link, self.res_1)
            self.split_high_res_dataset()

            if self.verbose:
                print("Dataset 1 not present, downloading it ...")

        if not os.path.exists(self.res_2_path):
            print("Dataset 1 not present, resizing it ...")
            self.resize_dataset(os.path.join(self.split, res_1), os.path.join(self.split, res_2), self.res_2_size)
        
        if not os.path.exists(self.res_3_path):
            print("Low dataset not present, resizing it ...")
            self.resize_dataset(os.path.join(self.split, res_1), os.path.join(self.split, res_3), self.res_3_size)
                    
        self.images_1 = os.listdir(self.res_1_path)
        self.images_2 = os.listdir(self.res_2_path)

    def get_link(self, res:str):
        with open("links.json") as f:
            data = json.load(f)
            try:
                return data["datasets"][res]
            except:
                raise KeyError(f"{res} dataset link not found")
            
    def open_image(self, path:str):
        #return np.array(Image.open(path))[:, :, 0:3]
        return cv2.imread(path)
            
    def look_for_dataset(self) -> bool:
        return 

    def unzip_file(self, file_path: str, extract_path: str):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    def download_dataset(self, url: str, dest: str):
        folder_path = os.path.join(self.resources_folder, dest)
        zip_path = folder_path + ".zip"

        if os.path.exists(folder_path):
            self.print(f"Dataset already downloaded at {folder_path}")
            return
        
        gdown.download(url, zip_path, quiet=False)

        self.print(f"Extracting to {folder_path} ...")
        self.unzip_file(zip_path, folder_path)
        
        os.remove(zip_path)
        
        self.print("Done!")

    def resize_dataset(self, source:str, dest:str, new_res:tuple) -> None:
        dir_source = os.path.join(self.resources_folder, source)
        dir_dest = os.path.join(self.resources_folder, dest)

        if not os.path.exists(dir_dest):
            os.makedirs(dir_dest)

        images = os.listdir(dir_source)

        self.print("Resizing images...")
        for img in images:
            source_img = os.path.join(dir_source, img)
            dest_img = os.path.join(dir_dest, img)
            print(new_res)
            self.resize_image(source_img, dest_img, new_res)
        self.print("Done!")

    def resize_image(self, source:str, dest:str, new_res:tuple) -> None:
        img = self.open_image(source)
        img = cv2.resize(img, new_res)
        cv2.imwrite(dest, img)

    def split_high_res_dataset(self) -> None:
        dir_train_high = os.path.join(self.resources_folder, self.train, self.res_1)
        dir_test_high = os.path.join(self.resources_folder, self.test, self.res_1)

        if not os.path.exists(dir_train_high):
            os.makedirs(dir_train_high)
        
        if not os.path.exists(dir_test_high):
            os.makedirs(dir_test_high)

        dir_high = os.path.join(self.resources_folder, self.res_1)
        images = os.listdir(dir_high)

        self.print("Moving images...")
        self._move_images(images[:int(0.8*len(images))], dir_high, dir_train_high)
        self._move_images(images[int(0.8*len(images)):], dir_high, dir_test_high)

        self._remove_folder(dir_high)
        self.print("Done!")

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
        return 3*len(self.images_1) if self.chosen_indices is None else len(self.chosen_indices)


    def __getitem__(self, index) -> Any:
        index = self.check_index(index)

        sub_index_category = index // len(self.images_1)
        sub_index = index % len(self.images_1)

        # Open image without transparence
        if sub_index_category == 0:
            high_res = self.open_image(os.path.join(self.res_1, self.images_1[index]))
            low_res = self.open_image(os.path.join(self.res_2, self.images_1[index]))
            scale_factor = self.upscale_factor_2
        elif sub_index_category == 1:
            high_res = self.open_image(os.path.join(self.res_1, self.images_1[index]))
            low_res = self.open_image(os.path.join(self.res_3, self.images_1[index]))
            scale_factor = self.upscale_factor_3
        else:
            high_res = self.open_image(os.path.join(self.res_2, self.images_1[index]))
            low_res = self.open_image(os.path.join(self.res_3, self.images_1[index]))
            scale_factor = self.upscale_factor_1

        if self.transforms is not None:
            high_res = self.transforms(high_res)
            low_res = self.transforms(low_res)
        
        return low_res, high_res, scale_factor

    @staticmethod
    def clean_dir(dir:str = "resources"):
        for d in os.listdir(dir):
            if os.path.isdir(d):
                os.rmdir(d)
    
    def get_info(self):
        # Display the sizes of the dataset
        dir_train_res_1 = os.path.join(self.resources_folder, self.train, self.res_1)
        dir_train_res_2 = os.path.join(self.resources_folder, self.train, self.res_2)
        dir_train_res_3 = os.path.join(self.resources_folder, self.train, self.res_3)
        dir_test_res_1 = os.path.join(self.resources_folder, self.test, self.res_1)
        dir_test_res_2 = os.path.join(self.resources_folder, self.test, self.res_2)
        dir_test_res_3 = os.path.join(self.resources_folder, self.test, self.res_3)

        self.print(f'Number of train res 1 images: {len(os.listdir(dir_train_res_1))}')
        self.print(f'Number of train res 2 resolution images: {len(os.listdir(dir_train_res_2))}')
        self.print(f'Number of train res 3 resolution images: {len(os.listdir(dir_train_res_3))}')
        self.print(f'Number of valid res 1 resolution images: {len(os.listdir(dir_test_res_1))}')
        self.print(f'Number of valid res 2 resolution images: {len(os.listdir(dir_test_res_2))}')
        self.print(f'Number of valid res 3 resolution images: {len(os.listdir(dir_test_res_3))}')


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision
    import numpy as np

    # CarlaDataset.clean_dir("resources")
    train = CarlaDatasetMultipleSize("1920x1080", "480x270", "240x135", transforms = torchvision.transforms.ToTensor(), download=True)
    # train.get_info()
    print(len(train))
    print(train[0][0].shape)
    print(train[0][1].shape)
    print(train[-1][0].shape)
    print(train[-1][1].shape)