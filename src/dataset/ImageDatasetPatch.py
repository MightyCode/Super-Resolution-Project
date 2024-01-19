from .ImageDataset import ImageDataset
from .PatchImageTool import PatchImageTool

from src.utils.PytorchUtil import PytorchUtil as torchUtil

import os
from typing import Any
import math

class ImageDatasetPatch(ImageDataset):
    def __init__(self, 
                 dataset_name: str = "train",
                 hr_name:str = "1920x1080", 
                 upscale_factors: list = [2], 
                 channels: list = ["r", "g", "b"],
                 transforms = None, 
                 download:bool = False, 
                 patch_size=16,
                 verbose:bool = True):
        super().__init__(dataset_name, hr_name, upscale_factors, channels, transforms, download, verbose=verbose)
        self.patch_sizes = []
        for upscale_factor in self.upscale_factors:
            self.patch_sizes.append(patch_size // upscale_factor)
            #self.patch_sizes.append(patch_size)

        self.number_patch_in_height = []
        self.number_patch_in_width = []

        self.number_patch_per_upscale = []

        self.total_number_patch = 0
        
        # If the image is not divisible by the patch size, we add a patch to the right and to the bottom
        for i in range(len(self.upscale_factors)):
            self.number_patch_in_width.append(math.ceil(self.lr_sizes[i][0] / self.patch_sizes[i]))
            self.number_patch_in_height.append(math.ceil(self.lr_sizes[i][1] / self.patch_sizes[i]))

            self.number_patch_per_upscale.append(self.number_patch_in_width[-1] * self.number_patch_in_height[-1])
        
        self.total_number_patch = self.number_patch_per_upscale[0]

    def __len__(self):
        if self.chosen_indices is not None:
            return len(self.chosen_indices)
        
        return super().__len__() * self.total_number_patch

    def get_number_patch_per_image(self, upscale_factor=None, upscale_index=None):
        if upscale_index is None:
            return self.number_patch_per_upscale[self.upscale_factors.index(upscale_factor)]
        
        return self.number_patch_per_upscale[upscale_index]
            
    
    def get_total_number_patch_per_image(self):
        return self.total_number_patch

    def get_patch_size(self, upscale_factor=None, upscale_index=None):
        if upscale_index is None:
            return self.patch_sizes[self.upscale_factors.index(upscale_factor)]
        
        return self.patch_sizes[upscale_index] 

    """
    Return the patch for all sub size
    """
    def __getitem__(self, index_patch) -> Any:
        index_patch = self.check_index(index_patch)

        image_index = index_patch // (self.total_number_patch)
        part_on_image = index_patch % (self.total_number_patch)

        hr_data_np = self.load_data_from_path(os.path.join(self.hr_path, self.images[image_index]))

        if self.transforms is None:
            hr_data_tensor = torchUtil.numpy_to_tensor(hr_data_np)
        else:
            hr_data_tensor = self.transforms(hr_data_np)
        
        hr_image_torch = self.filter_channels_to_image(hr_data_tensor)

        lr_data_patch_tensors = []

        for i, upscale_factor in enumerate(self.upscale_factors):
            lr_data_np = self.load_data_from_path(os.path.join(self.lr_paths[i], self.images[image_index]))
            
            if self.transforms is None:
                lr_data_tensor = torchUtil.numpy_to_tensor(lr_data_np)
            else:
                lr_data_tensor = self.transforms(lr_data_np)

            number_patch_width = self.number_patch_in_width[i]
            number_patch_height = self.number_patch_in_height[i]

            lr_data_patch_tensors.append(
                PatchImageTool.get_patch_from_image_index(
                    lr_data_tensor, 
                    part_on_image, self.patch_sizes[i], 
                    w=number_patch_width, h=number_patch_height)
            )
            
        hr_img_patch_tensor = PatchImageTool.get_patch_from_image_index(
            hr_image_torch, 
            part_on_image, self.patch_sizes[0] * self.upscale_factors[0],
            w=number_patch_width, h=number_patch_height)

        #upscale to torch
        return lr_data_patch_tensors, hr_img_patch_tensor


    def get_all_patch_for_image(self, index_patch, upscale_factor=None, upscale_index=None):
        if upscale_index is None and upscale_factor is None:
            raise Exception("You must specify the upscale factor or the upscale index")
        elif upscale_index is None: 
            upscale_index = self.upscale_factor_to_index(upscale_factor)

        index_patch = self.check_index(index_patch)

        number_patch_width = self.number_patch_in_width[upscale_index]
        number_patch_height = self.number_patch_in_height[upscale_index]

        image_index = index_patch // (self.total_number_patch)
        lr_data_np = self.load_data_from_path(os.path.join(self.lr_paths[upscale_index], self.images[image_index]))
        hr_data_np = self.load_data_from_path(os.path.join(self.hr_path, self.images[image_index]))

        if self.transforms is None:
            lr_data_tensor = torchUtil.numpy_to_tensor(lr_data_np)
            hr_data_tensor = torchUtil.numpy_to_tensor(hr_data_np)
        else:
            lr_data_tensor = self.transforms(lr_data_np)
            hr_data_tensor = self.transforms(hr_data_np)

        hr_img_tensor = self.filter_channels_to_image(hr_data_tensor)

        lr_data_patch_tensors: list = PatchImageTool.get_patchs_from_image(
            lr_data_tensor, self.patch_sizes[upscale_index], 
                w=number_patch_width, h=number_patch_height)

        hr_img_patch_tensors: list = PatchImageTool.get_patchs_from_image(
            hr_img_tensor, 
            self.patch_sizes[upscale_index] * self.upscale_factors[upscale_index], 
            w=number_patch_width, h=number_patch_height)
    
        return lr_data_patch_tensors, hr_img_patch_tensors
    
    def get_index_for_image(self, index_patch):
        index_patch = index_patch // (self.total_number_patch)

        if index_patch < 0:
            index_patch = len(self.images) + index_patch

        return index_patch

    def get_index_start_patch(self, index_patch):
        index_patch = index_patch // (self.total_number_patch)

        return index_patch * (self.total_number_patch)

    def get_full_image(self, index):
        return super().__getitem__(index)
    
    def get_low_res_full_image_size(self, upscale_factor=None, upscale_index=None):
        if upscale_index is None and upscale_factor is None:
            raise Exception("You must specify the upscale factor or the upscale index")
        elif upscale_index is None: 
            upscale_index = self.upscale_factor_to_index(upscale_factor)
            
        number_patch_width = self.number_patch_in_width[upscale_index]
        number_patch_height = self.number_patch_in_height[upscale_index]

        return (number_patch_height * self.patch_size, number_patch_width * self.patch_size)
    
    def print_info(self):
        super().print_info()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision
    import numpy as np

    lr = ImageDatasetPatch(dataset_name="train", 
                           hr_name="1920x1080",  
                            upscale_factors=[2, 4, 8],
                           transforms = torchvision.transforms.ToTensor(), 
                           download=False)
    
    lr.get_info()
    print(len(lr))
    print(lr[0][0].shape)
    print(lr[0][1].shape)
    print(lr[-1][0].shape)
    print(lr[-1][1].shape)

    lr.get_info()
    for i in range(3):
        low_res, high_res = lr[np.random.randint(len(lr))]
        #fig to show the 16x16 patch and the 32x32 patch
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(low_res.permute(1, 2, 0))
        fig.add_subplot(1, 2, 2)
        plt.imshow(high_res.permute(1, 2, 0))
        plt.show()