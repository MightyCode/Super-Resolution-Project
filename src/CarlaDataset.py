import os
from typing import Any
import gdown
import zipfile
from sklearn.model_selection import train_test_split
import json
from torch.utils.data import Dataset
import cv2

class CarlaDataset(Dataset):
    def __init__(self, high_res:str = "1920x1080", low_res:str = "1280x720", split:str = "train", transforms = None, download:bool = False):
        super().__init__()
        self.split = split
        self.resources_folder: str = "resources"
        self.high_res = high_res
        self.low_res = low_res
        self.train = "train"
        self.test = "test"
        self.transforms = transforms
        self.dataset_link = self.get_link(high_res)

        if self.look_for_dataset():
            print("Dataset already present")
            return
        if download:
            self.download_dataset(self.dataset_link, self.high_res)
    
        self.resize_dataset(self.high_res, self.low_res)
        self.split_dataset()


    def get_link(self, res:str):
        with open("links.json") as f:
            data = json.load(f)
            try:
                return data["datasets"][res]
            except:
                raise KeyError(f"{res} dataset link not found")
            
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
            source_img = cv2.imread(os.path.join(source_folder, img))
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

    def _remove_folder(self, folder_path: str) -> None:
        try:
            os.rmdir(folder_path)
            print(f"Folder '{folder_path}' removed successfully.")
        except OSError as e:
            print(f"Error: {e}")

    def __len__(self):
        return len(os.listdir(os.path.join(self.resources_folder, self.split, self.high_res)))

    def __getitem__(self, index) -> Any:
        dir_path = os.path.join(self.resources_folder, self.split)
        high_res_path = os.path.join(dir_path, self.high_res)
        low_res_path = os.path.join(dir_path, self.low_res)
        images = os.listdir(high_res_path)
        high_res = cv2.imread(os.path.join(high_res_path, images[index]))
        low_res = cv2.imread(os.path.join(low_res_path, images[index]))
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



class CarlaDataset16x16(CarlaDataset):
    def __init__(self, high_res:str = "1920x1080", low_res:str = "1280x720", split:str = "train", transforms = None, download:bool = False):
        super().__init__(high_res, low_res, split, transforms, download)
        #open the first image of the train low res set to get the size of the images
        I = cv2.imread(os.path.join(self.resources_folder, self.train, self.low_res, os.listdir(os.path.join(self.resources_folder, self.train, self.low_res))[0]))
        self.h = I.shape[0] // 16
        self.w = I.shape[1] // 16
        self.scale_factor = int(self.high_res.split("x")[0]) // int(self.low_res.split("x")[0])
    
    def __getitem__(self, index) -> Any:
        h = self.h
        w = self.w
        index = index // (h * w)

        dir_path = os.path.join(self.resources_folder, self.split)
        high_res_path = os.path.join(dir_path, self.high_res)
        low_res_path = os.path.join(dir_path, self.low_res)
        images = os.listdir(high_res_path)

        image_low_res = cv2.imread(os.path.join(low_res_path, images[index]))
        image_high_res = cv2.imread(os.path.join(high_res_path, images[index]))

        if self.transforms is not None:
            image_low_res = self.transforms(image_low_res)
            image_high_res = self.transforms(image_high_res)

        #take a 16x16 patch corresponding to index for the low res image and the 32x32 patch for the high res image
        i = index % (h * w)
        line = i // w
        col = i % h
        sf = self.scale_factor
        image_low_res = image_low_res[:, 16*line:16*(line+1), 16*col:16*(col+1)]
        image_high_res = image_high_res[:, 16*sf*line:16*sf*(line+1), 16*sf*col:16*sf*(col+1)]

        return image_low_res, image_high_res
    

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
    print(test[0][0].shape)
    print(test[0][1].shape)
    test.get_info()
    low_res, high_res = test[0]
    plt.imshow(low_res.permute(1, 2, 0))
    plt.show()
    plt.imshow(high_res.permute(1, 2, 0))
    plt.show()


    lr = CarlaDataset16x16(high_res="1920x1080", low_res="960x540", split="train", transforms = torchvision.transforms.ToTensor(), download=False)
    print(lr[0][0].shape)
    print(lr[0][1].shape)
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