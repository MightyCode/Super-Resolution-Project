import os
from typing import Any
import gdown
import zipfile
from sklearn.model_selection import train_test_split
import json
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np
import math

class CarlaDataset(Dataset):
    def __init__(self, 
                 high_res:str = "1920x1080", 
                 low_res:str = "1280x720", 
                 split:str = "train", 
                 transforms = None, 
                 download:bool = False):
        super().__init__()
        self.split = split
        self.resources_folder: str = "resources"
        self.high_res = high_res
        self.low_res = low_res
        self.train = "train"
        self.test = "test"
        self.transforms = transforms
        self.dataset_link = self.get_link(high_res)

        self.dir_path = os.path.join(self.resources_folder, self.split)
        self.high_res_path = os.path.join(self.dir_path, self.high_res)
        self.low_res_path = os.path.join(self.dir_path, self.low_res)

        self.chosen_indices = None

        if self.look_for_dataset():
            self.images = os.listdir(self.high_res_path)
            print("Dataset already present")

            return

        if download:
            self.download_dataset(self.dataset_link, self.high_res)
    
        self.resize_dataset(high_res, low_res)
        self.split_dataset()
                
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
        return os.path.exists(os.path.join(self.resources_folder, self.train)) or os.path.exists(os.path.join(self.resources_folder, self.test))

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

    def resize_dataset(self, source:str, dest:str) -> None:
        source_folder = os.path.join(self.resources_folder, source)
        dest_folder = os.path.join(self.resources_folder, dest)
        if os.path.exists(dest_folder):
            print("Dataset already resized")
            return
        os.mkdir(dest_folder)
        destx, desty = dest.split("x")
        print("Resizing images ...")

        for img in os.listdir(source_folder):
            source_img = cv2.imopen(os.path.join(source_folder, img))
            try:
                dest_img = cv2.resize(source_img,(int(destx), int(desty)))
                dest_img_path = os.path.join(dest_folder, img)
                cv2.imwrite(dest_img_path, dest_img)
            except:
                print(f"Broken image, skipping it ({img})")
                os.remove(os.path.join(source_folder, img))
        assert len(os.listdir(source_folder)) == len(os.listdir(dest_folder))
        print("Done!")

    def split_dataset(self) -> None:
        dir_train_high = os.path.join(self.resources_folder, self.train, self.high_res)
        dir_train_low = os.path.join(self.resources_folder, self.train, self.low_res)
        dir_test_high = os.path.join(self.resources_folder, self.test, self.high_res)
        dir_test_low = os.path.join(self.resources_folder, self.test, self.low_res)

        if os.path.exists(dir_train_high):
            print("Dataset already splitted")
            return
        
        os.makedirs(dir_train_high)
        os.makedirs(dir_train_low)
        os.makedirs(dir_test_high)
        os.makedirs(dir_test_low)

        dir_high = os.path.join(self.resources_folder, self.high_res)
        dir_low = os.path.join(self.resources_folder, self.low_res)
        images = os.listdir(dir_high)

        print("Moving images...")
        self._move_images(images[:int(0.8*len(images))], dir_high, dir_train_high)
        self._move_images(images[:int(0.8*len(images))], dir_low, dir_train_low)
        self._move_images(images[int(0.8*len(images)):], dir_high, dir_test_high)
        self._move_images(images[int(0.8*len(images)):], dir_low, dir_test_low)

        self._remove_folder(dir_high)
        self._remove_folder(dir_low)
        print("Done!")

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
            print(f"Folder '{folder_path}' removed successfully.")
        except OSError as e:
            print(f"Error: {e}")

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

        print(f'Number of train low resolution images: {len(os.listdir(dir_train_low))}')
        print(f'Number of train high resolution images: {len(os.listdir(dir_train_high))}')
        print(f'Number of valid low resolution images: {len(os.listdir(dir_test_low))}')
        print(f'Number of valid high resolution images: {len(os.listdir(dir_test_high))}')



class CarlaDatasetPatch(CarlaDataset):
    def __init__(self, 
                 high_res:str = "1920x1080", 
                 low_res:str = "1280x720", 
                 split:str = "train", 
                 transforms = None, 
                 download:bool = False, 
                 patch_size=16):
        super().__init__(high_res, low_res, split, transforms, download)
        self.patch_size = patch_size
        #open the first image of the train low res set to get the size of the images
        I = self.open_image(os.path.join(self.resources_folder, self.train, self.low_res,
                                        os.listdir(os.path.join(self.resources_folder, self.train, self.low_res))[0]))  
        
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

        #take a 16x16 patch corresponding to index for the low res image and the 32x32 patch for the high res image
        line = part_on_image % self.h
        col = part_on_image // self.h
        sf = self.scale_factor

        start_low_x = self.patch_size * line
        start_low_y = self.patch_size * col

        end_low_x = start_low_x + self.patch_size
        end_low_y = start_low_y + self.patch_size

        if end_low_x > image_low_res.shape[1]:
            end_low_x = image_low_res.shape[1]
            start_low_x = end_low_x - self.patch_size

        if end_low_y > image_low_res.shape[2]:
            end_low_y = image_low_res.shape[2]
            start_low_y = end_low_y - self.patch_size


        start_high_x = start_low_x * sf
        start_high_y = start_low_y * sf

        end_high_x = end_low_x * sf
        end_high_y = end_low_y * sf

        image_low_res = image_low_res[:, start_low_x: end_low_x, start_low_y : end_low_y]
        image_high_res = image_high_res[:, start_high_x: end_high_x, start_high_y : end_high_y]

        return image_low_res, image_high_res

    def get_all_patch_for_image(self, index_patch):
        image_index = index_patch // (self.h * self.w)
        image_low_res = self.open_image(os.path.join(self.low_res_path, self.images[image_index]))
        image_high_res = self.open_image(os.path.join(self.high_res_path, self.images[image_index]))

        if self.transforms is not None:
            image_low_res = self.transforms(image_low_res)
            image_high_res = self.transforms(image_high_res)

        patches_low_res = np.zeros((self.h * self.w, 3, self.patch_size, self.patch_size))
        patches_high_res = np.zeros((self.h * self.w, 3, self.patch_size * self.scale_factor, self.patch_size * self.scale_factor))

        for i in range(self.h):
            for j in range(self.w):
                sf = self.scale_factor
                start_low_x = self.patch_size * i
                start_low_y = self.patch_size * j

                end_low_x = start_low_x + self.patch_size
                end_low_y = start_low_y + self.patch_size

                if end_low_x > image_low_res.shape[1]:
                    end_low_x = image_low_res.shape[1]
                    start_low_x = end_low_x - self.patch_size

                if end_low_y > image_low_res.shape[2]:
                    end_low_y = image_low_res.shape[2]
                    start_low_y = end_low_y - self.patch_size


                start_high_x = start_low_x * sf
                start_high_y = start_low_y * sf

                end_high_x = end_low_x * sf
                end_high_y = end_low_y * sf

                patches_low_res[i * self.w + j] = image_low_res[:, start_low_x: end_low_x, start_low_y : end_low_y]
                patches_high_res[i * self.w + j] = image_high_res[:, start_high_x: end_high_x, start_high_y : end_high_y]

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
    
    def get_info(self):
        # Display the sizes of the dataset
        dir_train_high = os.path.join(self.resources_folder, self.train, self.high_res)
        dir_train_low = os.path.join(self.resources_folder, self.train, self.low_res)
        dir_test_high = os.path.join(self.resources_folder, self.test, self.high_res)
        dir_test_low = os.path.join(self.resources_folder, self.test, self.low_res)

        print(f'Number of train low resolution images: {len(os.listdir(dir_train_low)) * self.h * self.w}')
        print(f'Number of train high resolution images: {len(os.listdir(dir_train_high)) * self.h * self.w}')
        print(f'Number of valid low resolution images: {len(os.listdir(dir_test_low)) * self.h * self.w}')
        print(f'Number of valid high resolution images: {len(os.listdir(dir_test_high)) * self.h * self.w}')

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