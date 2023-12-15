from src.PatchImageTool import PatchImageTool

import os
from typing import Any
import gdown
import zipfile
import json
from torch.utils.data import Dataset
import cv2
import numpy as np
import math

class CarlaDataset(Dataset):
    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def __init__(self, 
                 high_res:str = "1920x1080", 
                 low_res:str = "1280x720", 
                 split:str = "train", 
                 transforms = None, 
                 download:bool = False,
                 verbose:bool = True):
        super().__init__()
        
        self.split = split
        self.resources_folder: str = "resources"

        self.high_res = high_res
        self.low_res = low_res

        self.train = "train"
        self.test = "test"

        temp = high_res.split("x")
        self.high_res_size = (int(temp[0]), int(temp[1]))

        temp = low_res.split("x")
        self.low_res_size = (int(temp[0]), int(temp[1]))

        self.upscale_factor = int(self.high_res.split("x")[0]) / int(self.low_res.split("x")[0])

        self.transforms = transforms
        self.dataset_link = self.get_link(high_res)

        self.dir_path = os.path.join(self.resources_folder, self.split)
        self.high_res_path = os.path.join(self.dir_path, self.high_res)
        self.low_res_path = os.path.join(self.dir_path, self.low_res)

        self.chosen_indices = None
        
        self.verbose = verbose
        
        if self.verbose:
            self.print("Check for download and resize ...")

        if download and not os.path.exists(self.high_res_path):
            self.download_dataset(self.dataset_link, self.high_res)
            self.split_high_res_dataset()

            if self.verbose:
                print("High dataset not present, downloading it ...")

        if not os.path.exists(self.low_res_path):
            print("Low dataset not present, resizing it ...")
            self.resize_dataset(os.path.join(self.split, high_res), os.path.join(self.split, low_res))
                    
        self.images = os.listdir(self.high_res_path)

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

    def resize_dataset(self, source:str, dest:str) -> None:
        dir_source = os.path.join(self.resources_folder, source)
        dir_dest = os.path.join(self.resources_folder, dest)

        if not os.path.exists(dir_dest):
            os.makedirs(dir_dest)

        images = os.listdir(dir_source)

        self.print("Resizing images...")
        for img in images:
            source_img = os.path.join(dir_source, img)
            dest_img = os.path.join(dir_dest, img)
            self.resize_image(source_img, dest_img)
        self.print("Done!")

    def resize_image(self, source:str, dest:str) -> None:
        img = self.open_image(source)
        img = cv2.resize(img, self.low_res_size)
        cv2.imwrite(dest, img)

    def split_high_res_dataset(self) -> None:
        dir_train_high = os.path.join(self.resources_folder, self.train, self.high_res)
        dir_test_high = os.path.join(self.resources_folder, self.test, self.high_res)

        if not os.path.exists(dir_train_high):
            os.makedirs(dir_train_high)
        
        if not os.path.exists(dir_test_high):
            os.makedirs(dir_test_high)

        dir_high = os.path.join(self.resources_folder, self.high_res)
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
        return len(self.images) if self.chosen_indices is None else len(self.chosen_indices)


    def __getitem__(self, index) -> Any:
        index = self.check_index(index)

        # Open image without transparence
        high_res = self.open_image(os.path.join(self.high_res_path, self.images[index]))
        low_res = self.open_image(os.path.join(self.low_res_path, self.images[index]))

        if self.transforms is not None:
            high_res = self.transforms(high_res)
            low_res = self.transforms(low_res)
        
        return low_res, high_res

    @staticmethod
    def clean_dir(dir:str = "resources"):
        for d in os.listdir(dir):
            if os.path.isdir(d):
                os.rmdir(d)
    
    def get_info(self):
        # Display the sizes of the dataset
        dir_train_high = os.path.join(self.resources_folder, self.train, self.high_res)
        dir_train_low = os.path.join(self.resources_folder, self.train, self.low_res)
        dir_test_high = os.path.join(self.resources_folder, self.test, self.high_res)
        dir_test_low = os.path.join(self.resources_folder, self.test, self.low_res)

        self.print(f'Number of train low resolution images: {len(os.listdir(dir_train_low))}')
        self.print(f'Number of train high resolution images: {len(os.listdir(dir_train_high))}')
        self.print(f'Number of valid low resolution images: {len(os.listdir(dir_test_low))}')
        self.print(f'Number of valid high resolution images: {len(os.listdir(dir_test_high))}')



class CarlaDatasetPatch(CarlaDataset):
    def __init__(self, 
                 high_res:str = "1920x1080", 
                 low_res:str = "1280x720", 
                 split:str = "train", 
                 transforms = None, 
                 download:bool = False, 
                 patch_size=16,
                 verbose:bool = True):
        super().__init__(high_res, low_res, split, transforms, download, verbose=verbose)
        self.patch_size = patch_size
        #open the first image of the train low res set to get the size of the images
        I = self.open_image(os.path.join(self.resources_folder, self.train, self.low_res,
                                        os.listdir(os.path.join(self.resources_folder, self.train, self.low_res))[0]))  
        
        self.low_res_image_size = I.shape[0:2]


        # If the image is not divisible by the patch size, we add a patch to the right and to the bottom
        self.h = math.ceil(I.shape[0] / self.patch_size)
        self.w = math.ceil(I.shape[1] / self.patch_size)

        self.scale_factor = int(self.high_res.split("x")[0]) // int(self.low_res.split("x")[0])
    
    def get_number_patch_per_image(self):
        return self.h * self.w

    def __len__(self):
        if self.chosen_indices is not None:
            return len(self.chosen_indices)
        
        return super().__len__() * self.h * self.w

    def __getitem__(self, index_patch) -> Any:
        index_patch = self.check_index(index_patch)
        
        image_index = index_patch // (self.h * self.w)
        part_on_image = index_patch % (self.h * self.w)

        image_low_res = self.open_image(os.path.join(self.low_res_path, self.images[image_index]))
        image_high_res = self.open_image(os.path.join(self.high_res_path, self.images[image_index]))

        if self.transforms is not None:
            image_low_res = self.transforms(image_low_res)
            image_high_res = self.transforms(image_high_res)

        patch_low_res = PatchImageTool.get_patch_from_image_index(image_low_res, part_on_image, self.patch_size, w=self.w, h=self.h)
        patch_high_res = PatchImageTool.get_patch_from_image_index(image_high_res, part_on_image, self.patch_size * self.scale_factor, w=self.w, h=self.h)

        return patch_low_res, patch_high_res

    def get_all_patch_for_image(self, index_patch):
        image_index = index_patch // (self.h * self.w)
        image_low_res = self.open_image(os.path.join(self.low_res_path, self.images[image_index]))
        image_high_res = self.open_image(os.path.join(self.high_res_path, self.images[image_index]))

        if self.transforms is not None:
            image_low_res = self.transforms(image_low_res)
            image_high_res = self.transforms(image_high_res)

        patches_low_res = PatchImageTool.get_patchs_from_image(image_low_res, self.patch_size, w=self.w, h=self.h)
        patches_high_res = PatchImageTool.get_patchs_from_image(image_high_res, self.patch_size * self.scale_factor, w=self.w, h=self.h)
    
        return patches_low_res, patches_high_res
    
    def get_index_for_image(self, index_patch):
        index_patch = index_patch // (self.h * self.w)

        if index_patch < 0:
            index_patch = len(self.images) + index_patch

        return index_patch

    def get_index_start_patch(self, index_patch):
        index_patch = index_patch // (self.h * self.w)

        return index_patch * (self.h * self.w)

    def get_full_image(self, index):
        return super().__getitem__(index)
    
    def get_low_res_full_image_size(self):
        return (self.h * self.patch_size, self.w * self.patch_size)
    
    def get_info(self):
        # Display the sizes of the dataset
        dir_train_high = os.path.join(self.resources_folder, self.train, self.high_res)
        dir_train_low = os.path.join(self.resources_folder, self.train, self.low_res)
        dir_test_high = os.path.join(self.resources_folder, self.test, self.high_res)
        dir_test_low = os.path.join(self.resources_folder, self.test, self.low_res)

        self.print(f'Number of train low resolution images: {len(os.listdir(dir_train_low)) * self.h * self.w}')
        self.print(f'Number of train high resolution images: {len(os.listdir(dir_train_high)) * self.h * self.w}')
        self.print(f'Number of valid low resolution images: {len(os.listdir(dir_test_low)) * self.h * self.w}')
        self.print(f'Number of valid high resolution images: {len(os.listdir(dir_test_high)) * self.h * self.w}')



class CarlaDatasetMultiplePatch(CarlaDataset):
    '''This dataset is used to train the model with multiple patchs.
    You should have 3 input sizes, the high res should be 8_th time higher than the low res 1 and 4_th time higher than the low res 2'''
    def __init__(self, 
                 high_res:str = "1920x1080", 
                 low_res_1:str = "1280x720",
                 low_res_2:str = "960x540", 
                 split:str = "train", 
                 transforms = None, 
                 download:bool = False,
                 patch_size=16,
                 verbose:bool = True):
        super().__init__(high_res, low_res_1, split, transforms, download, verbose=verbose)
        self.patch_size_1 = patch_size
        self.low_res_2 = low_res_2
        self.low_res_path_2 = os.path.join(self.dir_path, self.low_res_2)
        #open the first image of the train low res set to get the size of the images
        I = self.open_image(os.path.join(self.resources_folder, self.train, self.low_res_1,
                                        os.listdir(os.path.join(self.resources_folder, self.train, self.low_res_1))[0]))  
        
        self.low_res_image_size_1 = I.shape[0:2]

        # If the image is not divisible by the patch size, we add a patch to the right and to the bottom
        self.h = math.ceil(I.shape[0] / self.patch_size_1)
        self.w = math.ceil(I.shape[1] / self.patch_size_1)

        I = self.open_image(os.path.join(self.resources_folder, self.train, self.low_res_2,
                                        os.listdir(os.path.join(self.resources_folder, self.train, self.low_res_2))[0]))  
        
        self.low_res_image_size_2 = I.shape[0:2]

        self.scale_factor_1 = int(self.high_res.split("x")[0]) // int(self.low_res_1.split("x")[0])
        self.scale_factor_2 = int(self.low_res_2.split("x")[0]) // int(self.low_res_1.split("x")[0])
        self.scale_factor_3 = int(self.high_res.split("x")[0]) // int(self.low_res_2.split("x")[0])
    
    def get_number_patch_per_image(self):
        return self.h * self.w

    def __len__(self):
        if self.chosen_indices is not None:
            return len(self.chosen_indices)
        
        return super().__len__() * self.h * self.w

    def __getitem__(self, index_patch) -> Any:
        index_patch = self.check_index(index_patch)
        
        image_index = index_patch // (self.h * self.w)
        part_on_image = index_patch % (self.h * self.w)

        image_low_res = self.open_image(os.path.join(self.low_res_path, self.images[image_index]))
        image_high_res = self.open_image(os.path.join(self.high_res_path, self.images[image_index]))

        if self.transforms is not None:
            image_low_res = self.transforms(image_low_res)
            image_high_res = self.transforms(image_high_res)

        patch_low_res = PatchImageTool.get_patch_from_image_index(image_low_res, part_on_image, self.patch_size, w=self.w, h=self.h)
        patch_high_res = PatchImageTool.get_patch_from_image_index(image_high_res, part_on_image, self.patch_size * self.scale_factor, w=self.w, h=self.h)

        return patch_low_res, patch_high_res

    def get_all_patch_for_image(self, index_patch):
        image_index = index_patch // (self.h * self.w)
        image_low_res = self.open_image(os.path.join(self.low_res_path, self.images[image_index]))
        image_high_res = self.open_image(os.path.join(self.high_res_path, self.images[image_index]))

        if self.transforms is not None:
            image_low_res = self.transforms(image_low_res)
            image_high_res = self.transforms(image_high_res)

        patches_low_res = PatchImageTool.get_patchs_from_image(image_low_res, self.patch_size, w=self.w, h=self.h)
        patches_high_res = PatchImageTool.get_patchs_from_image(image_high_res, self.patch_size * self.scale_factor, w=self.w, h=self.h)
    
        return patches_low_res, patches_high_res
    
    def get_index_for_image(self, index_patch):
        index_patch = index_patch // (self.h * self.w)

        if index_patch < 0:
            index_patch = len(self.images) + index_patch

        return index_patch

    def get_index_start_patch(self, index_patch):
        index_patch = index_patch // (self.h * self.w)

        return index_patch * (self.h * self.w)

    def get_full_image(self, index):
        return super().__getitem__(index)
    
    def get_low_res_full_image_size(self):
        return (self.h * self.patch_size, self.w * self.patch_size)
    
    def get_info(self):
        # Display the sizes of the dataset
        dir_train_high = os.path.join(self.resources_folder, self.train, self.high_res)
        dir_train_low = os.path.join(self.resources_folder, self.train, self.low_res)
        dir_test_high = os.path.join(self.resources_folder, self.test, self.high_res)
        dir_test_low = os.path.join(self.resources_folder, self.test, self.low_res)

        self.print(f'Number of train low resolution images: {len(os.listdir(dir_train_low)) * self.h * self.w}')
        self.print(f'Number of train high resolution images: {len(os.listdir(dir_train_high)) * self.h * self.w}')
        self.print(f'Number of valid low resolution images: {len(os.listdir(dir_test_low)) * self.h * self.w}')
        self.print(f'Number of valid high resolution images: {len(os.listdir(dir_test_high)) * self.h * self.w}')

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision
    import numpy as np
    
    # CarlaDataset.clean_dir("resources")
    test = CarlaDataset("1920x1080", "960x540", transforms = torchvision.transforms.ToTensor(), download=False)
    test.get_info()
    print(len(test))
    print(test[0][0].shape)
    print(test[0][1].shape)
    print(test[-1][0].shape)
    print(test[-1][1].shape)
    
    """test.get_info()
    low_res, high_res = test[0]
    plt.imshow(low_res.permute(1, 2, 0))
    plt.show()
    plt.imshow(high_res.permute(1, 2, 0))
    plt.show()"""


    lr = CarlaDatasetPatch(high_res="1920x1080", 
                           low_res="960x540", 
                           split="train", 
                           transforms = torchvision.transforms.ToTensor(), 
                           download=False)
    
    lr.get_info()
    print(len(lr))
    print(lr[0][0].shape)
    print(lr[0][1].shape)
    print(lr[-1][0].shape)
    print(lr[-1][1].shape)

    """lr.get_info()
    for i in range(3):
        low_res, high_res = lr[np.random.randint(len(lr))]
        #fig to show the 16x16 patch and the 32x32 patch
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(low_res.permute(1, 2, 0))
        fig.add_subplot(1, 2, 2)
        plt.imshow(high_res.permute(1, 2, 0))
        plt.show()"""