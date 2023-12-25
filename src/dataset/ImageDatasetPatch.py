from .ImageDataset import ImageDataset
from .PatchImageTool import PatchImageTool

import os
from typing import Any
import math
import torch

class ImageDatasetPatch(ImageDataset):
    def __init__(self, 
                 dataset_name: str = "train",
                 high_res:str = "1920x1080", 
                 upscale_factors: list = [2], 
                 transforms = None, 
                 download:bool = False, 
                 patch_size=16,
                 verbose:bool = True):
        super().__init__(dataset_name, high_res, upscale_factors, transforms, download, verbose=verbose)
        self.patch_sizes = []
        for upscale_factor in self.upscale_factors:
            self.patch_sizes.append(patch_size // upscale_factor)
            #self.patch_sizes.append(patch_size)

        self.number_patch_heights = []
        self.number_patch_widths = []

        self.number_patch_per_upscale = []

        self.total_number_patch = 0
        
        # If the image is not divisible by the patch size, we add a patch to the right and to the bottom
        for i in range(len(self.upscale_factors)):
            self.number_patch_widths.append(math.ceil(self.low_res_sizes[i][0] / self.patch_sizes[i]))
            self.number_patch_heights.append(math.ceil(self.low_res_sizes[i][1] / self.patch_sizes[i]))

            self.number_patch_per_upscale.append(self.number_patch_widths[-1] * self.number_patch_heights[-1])

        
        self.total_number_patch = self.number_patch_per_upscale[0]

    def __len__(self):
        if self.chosen_indices is not None:
            return len(self.chosen_indices)
        
        return super().__len__() * self.total_number_patch

    def get_number_patch_per_image(self, upscale_factor=None, upscale_index=None):
        if upscale_index is not None:
            return self.number_patch_per_upscale[upscale_index]
        else:
            return self.number_patch_per_upscale[self.upscale_factors.index(upscale_factor)]
    
    def get_total_number_patch_per_image(self):
        return self.total_number_patch

    def get_patch_size(self, upscale_factor=None, upscale_index=None):
        if upscale_index is not None:
            return self.patch_sizes[upscale_index]
        
        return self.patch_sizes[self.upscale_factors.index(upscale_factor)]

    """
    Return the patch for all sub size
    """
    def __getitem__(self, index_patch) -> Any:
        index_patch = self.check_index(index_patch)

        image_index = index_patch // (self.total_number_patch)
        part_on_image = index_patch % (self.total_number_patch)

        image_high_res = self.open_image(os.path.join(self.high_res_path, self.images[image_index]))

        if self.transforms is not None:
            image_high_res = self.transforms(image_high_res)

        patchs_low_res = []

        for i, upscale_factor in enumerate(self.upscale_factors):
            image_low_res = self.open_image(os.path.join(self.low_res_paths[i], self.images[image_index]))
            
            if self.transforms is not None:
                image_low_res = self.transforms(image_low_res)

            number_patch_width = self.number_patch_widths[i]
            number_patch_height = self.number_patch_heights[i]

            patchs_low_res.append(
                PatchImageTool.get_patch_from_image_index(
                    image_low_res, 
                    part_on_image, self.patch_sizes[i], 
                    w=number_patch_width, h=number_patch_height)
            )
            
        patch_high_res = PatchImageTool.get_patch_from_image_index(
            image_high_res, 
            part_on_image, self.patch_sizes[0] * self.upscale_factors[0],
            w=number_patch_width, h=number_patch_height)

        #upscale to torch
        return patchs_low_res, patch_high_res


    def get_all_patch_for_image(self, index_patch, upscale_factor=None, upscale_index=None):
        if upscale_index is None and upscale_factor is None:
            raise Exception("You must specify the upscale factor or the upscale index")
        elif upscale_index is None: 
            upscale_index = self.upscale_factor_to_index(upscale_factor)

        index_patch = self.check_index(index_patch)

        number_patch_width = self.number_patch_widths[upscale_index]
        number_patch_height = self.number_patch_heights[upscale_index]

        image_index = index_patch // (self.total_number_patch)
        image_low_res = self.open_image(os.path.join(self.low_res_paths[upscale_index], self.images[image_index]))
        image_high_res = self.open_image(os.path.join(self.high_res_path, self.images[image_index]))

        if self.transforms is not None:
            image_low_res = self.transforms(image_low_res)
            image_high_res = self.transforms(image_high_res)

        patches_low_res = PatchImageTool.get_patchs_from_image(
            image_low_res, self.patch_sizes[upscale_index], 
                w=number_patch_width, h=number_patch_height)

        patches_high_res = PatchImageTool.get_patchs_from_image(
            image_high_res, 
            self.patch_sizes[upscale_index] * self.upscale_factors[upscale_index], 
            w=number_patch_width, h=number_patch_height)
    
    
        return patches_low_res, patches_high_res
    
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
            
        number_patch_width = self.number_patch_widths[upscale_index]
        number_patch_height = self.number_patch_heights[upscale_index]

        return (number_patch_height * self.patch_size, number_patch_width * self.patch_size)
    
    def print_info(self):
        super().print_info()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision
    import numpy as np

    lr = ImageDatasetPatch(dataset_name="train", 
                           high_res="1920x1080",  
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