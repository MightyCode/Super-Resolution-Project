# Super-Resolution-Project
S9 enseirb IA project for super resolution

## Dataset 1 : images (RGB)

| Training type / model | Train PSNR-SSIM x2 | Train  PSNR-SSIM x4 | Train PSNR-SSIM x8 | Test PSNR-SSIM x2 | Test  PSNR-SSIM x4 | Test PSNR-SSIM x8 |
| --------------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Nearsest | 33.80 - 0.917 | 28.58 - 0.788 | 24.66 - 0.640 | 33.61 - 0.913 | 28.44 - 0.78 | 24.59 - 0.636 |
| Billinear | 35.02 - 0.918 | 30.33 - 0.820 | 26.33 - 0.701 |  34.76 - 0.913 | 30.16 - 0.814 | 26.25 - 0.697 |
| Bicubic | 36.53 - 0.935 | 30.70 - 0.830 | 26.25 - 0.698  | 36.20 - 0.931 | 30.50 - 0.823 | 26.17 - 0.694  |
| Reference model | X - X | X - X | X - X | X - X | X - X | X - X |
| Upscale residual lpips - image - x2 | 38.04 - 0.943 | 30.62 - 0.828 | 26.23 - 0.697  | 37.65 - 0.938 | 30.34 - 0.820 | 26.14 - 0.693 |
| Upscale residual lpips - image - x2 x4 x8  | 38.81 - 0.953 | 32.13 - 0.86 | 27.48 - 0.738 | 38.80 - 0.962 | 32.54 - 0.893 | 28.10 - 0.800 |

## Dataset 2 : images (RGBDS)

D is for Depth, S is for segmentation.

| Training type / model | Train PSNR-SSIM x2 | Train  PSNR-SSIM x4 | Train PSNR-SSIM x8 | Test PSNR-SSIM x2 | Test  PSNR-SSIM x4 | Test PSNR-SSIM x8 |
| --------------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Nearsest | 36.79 - 0.964 | 33.09 - 0.905 | 28.68 - 0.822 | 36.69 - 0.966 | 32.03 - 0.911 | 28.67 - 0.835 |
| Billinear | 38.08 - 0.968 | 33.25 - 0.919 | 29.53 - 0.854 |  37.94 - 0.967 | 33.19 - 0.925 | 39.57. - 0.866 |
| Bicubic | 39.45 - 0.975 | 34.08 - 0.928 | 30.12 - 0.860  | 39.25 - 0.976 | 33.95 - 0.933 | 30.13 - 0.872  |
| Reference model | X - X | X - X | X - X | X - X | X - X | X - X |
| Upscale residual lpips - image - x2 | 40.85 - 0.977 | 34.16 - 0.930 | 30.13 - 0.860 | 40.66 - 0.979 | 34.02 - 0.933 | 30.13 - 0.871 |
| Upscale residual lpips - image - x2 x4 x8  | 41.19 - 0.979  | 34.35 - 0.931 | 30.15 - 0.861 | 41.05 - 0.980 | 34.17 - 0.936 | 30.15 - 0.872 |
| Model - (image/patch) - x2 x4 x8 - Carla features | 40.15 - 0.974 | 34.65 - 0.932 | 30.19 - 0.861 | 39.86 - 0.976 | 34.42 - 0.937 | 30.18 - 0.872 |

To obtain these results we used area down resolution methods (in order to create the dataset of smaller images) b : area, g : area, r : area, d : bicubic, s : bicubic. 


## Installation

Version of python we used to lauch the code: 
```
3.10.13
```

To install the libraries, use the requirements.txt

## Data set construction

Version of python we used to lauch carla : 
```
3.8
```
### Carla installation

To install Carla Simulator, follow the instructions on the official website : https://carla.readthedocs.io/en/latest/start_quickstart/

### Carla data set construction

To create the dataset, first launch the Carla server as expained on the official website :
https://carla.readthedocs.io/en/latest/start_quickstart/#running-carla

Then, launch the script to create the dataset :
```bash
python3 carla_dataset_creation.py
```

## Usage

### Type and size of data

The BGR format or derivatives (like BGRDS) is and must be always used. It should be converted to RBG only when displaying the data.

Tensor sizes ((size of batch), number of channels, height, width)
Numpy array size ((size of batch), width, height, number of channels)

Data number of channels is a minimum of 3 and can be more.
Image number of channems is 3.

### Data naming

hr => data or image of high resolution
lr => data or image low resolution

_data => entry data that is getable using operation [X] where X is an index.
_img => then when we are sure that the data is converted to a 3 channel image, we name the associated variable as stated.

In fact our datasets return all the low resolutions data and the associated high resolution data.

_patch => if the data is a patch of a higher data.

_tensor => if the data is a tensor
_np => if the data is numpy array
 => if the data is 

+(s) if the data is a list 

**Examples*

hr_data_patch_tensors => a list of data patches represented as a tensor.

lr_img_np => an image presented as numpy array.


## Folders

### Resources

All the datasets, used images and json data used for tests purpose should be placed here.

### Results

All the weights, the upscaled images and created information for tests purpose will be saved here.

Weights : 
```
weights-upscale-residual-lpips-v.2 : our best model, BGR model, uses rgb images
weights-upscale-residual2-lpips-v.4.1 : best BGRDS model, uses numpy array as images
```