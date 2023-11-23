import os
import zipfile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset

base_dir = '../resources/carla'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')

# Directories with training/validation low_res/high_res pictures
train_low_res_dir = os.path.join(train_dir, 'low_res')
train_high_res_dir = os.path.join(train_dir, 'high_res')
valid_low_res_dir = os.path.join(valid_dir, 'low_res')
valid_high_res_dir = os.path.join(valid_dir, 'high_res')

class LowAndHighResImageDataset(Dataset):
    def __init__(self, transform, train):
        self.transform = transform
        self.train = train
        self.images = []
        if self.train:
            self.low_res_folder = train_low_res_dir
            self.high_res_folder = train_high_res_dir
        else:
            self.low_res_folder = valid_low_res_dir
            self.high_res_folder = valid_high_res_dir

        self.low_res_images = [os.path.join(self.low_res_folder, img) for img in os.listdir(self.low_res_folder)]
        self.high_res_images = [os.path.join(self.high_res_folder, img) for img in os.listdir(self.high_res_folder)]
        self.low_res_images.sort()
        self.high_res_images.sort()
        assert len(self.low_res_images) == len(self.high_res_images)

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        img_low_res_path = self.low_res_images[idx]
        img_high_res_path = self.high_res_images[idx]
        image_low_res = Image.open(img_low_res_path).convert("RGB")
        image_high_res = Image.open(img_high_res_path).convert("RGB")
        if self.transform:
            image_low_res = self.transform(image_low_res)
            image_high_res = self.transform(image_high_res)

        return image_low_res, image_high_res
    
    def get_info(self):
        # Display the sizes of our dataset
        print(f'Number of train low resolution images: {len(os.listdir(train_low_res_dir))}')
        print(f'Number of train high resolution images: {len(os.listdir(train_high_res_dir))}')
        print(f'Number of valid low resolution images: {len(os.listdir(valid_low_res_dir))}')
        print(f'Number of valid high resolution images: {len(os.listdir(valid_high_res_dir))}')




class LowAndHighRes16x16ImageDataset(Dataset):
    def __init__(self, train, transform=transforms.ToTensor(), scale_factor=2):
        self.transform = transform
        self.train = train
        self.scale_factor = scale_factor
        self.images = []
        if self.train:
            self.low_res_folder = train_low_res_dir
            self.high_res_folder = train_high_res_dir
        else:
            self.low_res_folder = valid_low_res_dir
            self.high_res_folder = valid_high_res_dir

        self.low_res_images = [os.path.join(self.low_res_folder, img) for img in os.listdir(self.low_res_folder)]
        self.high_res_images = [os.path.join(self.high_res_folder, img) for img in os.listdir(self.high_res_folder)]
        self.low_res_images.sort()
        self.high_res_images.sort()
        I = Image.open(self.low_res_images[0]).convert("RGB")
        self.h = I.height // 16
        self.w = I.width // 16
        assert len(self.low_res_images) == len(self.high_res_images)

    def __len__(self):
        return len(self.low_res_images) * self.h * self.w

    def __getitem__(self, idx):
        h = self.h
        w = self.w
        idx = idx // (h * w)
        img_low_res_path = self.low_res_images[idx]
        img_high_res_path = self.high_res_images[idx]
        image_low_res = Image.open(img_low_res_path).convert("RGB")
        image_high_res = Image.open(img_high_res_path).convert("RGB")

        image_low_res = self.transform(image_low_res)
        image_high_res = self.transform(image_high_res)

        #take a 16x16 patch corresponding to idx for the low res image and the 32x32 patch for the high res image
        i = idx % (h * w)
        line = i // w
        col = i % h
        sf = self.scale_factor
        image_low_res = image_low_res[:, 16*line:16*(line+1), 16*col:16*(col+1)]
        image_high_res = image_high_res[:, 16*sf*line:16*sf*(line+1), 16*sf*col:16*sf*(col+1)]

        return image_low_res, image_high_res
    
    def get_info(self):
        # Display the sizes of our dataset
        print(f'Number of train low resolution images: {len(os.listdir(train_low_res_dir)) * self.h * self.w}')
        print(f'Number of train high resolution images: {len(os.listdir(train_high_res_dir)) * self.h * self.w}')
        print(f'Number of valid low resolution images: {len(os.listdir(valid_low_res_dir)) * self.h * self.w}')
        print(f'Number of valid high resolution images: {len(os.listdir(valid_high_res_dir)) * self.h * self.w}')


if __name__ == '__main__':
    lr = LowAndHighResImageDataset(transform=torchvision.transforms.ToTensor(), train=True)
    print(lr[0][0].shape)
    print(lr[0][1].shape)
    lr.get_info()
    low_res, high_res = lr[0]
    plt.imshow(low_res.permute(1, 2, 0))
    plt.show()
    plt.imshow(high_res.permute(1, 2, 0))
    plt.show()


    lr = LowAndHighRes16x16ImageDataset(transform=torchvision.transforms.ToTensor(), train=True)
    print(lr[0][0].shape)
    print(lr[0][1].shape)
    lr.get_info()
    for i in range(5):
        low_res, high_res = lr[np.random.randint(len(lr))]
        #fig to show the 16x16 patch and the 32x32 patch
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(low_res.permute(1, 2, 0))
        fig.add_subplot(1, 2, 2)
        plt.imshow(high_res.permute(1, 2, 0))
        plt.show()
    