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

The project is guaranteed to work with the folowing versions of python (other may work, but are not guaranteed):
```
3.10.13
3.8
```

To install the libraries, use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Data set construction

The carla simulator works on the following python version: 
```
3.8
```
### Carla installation

To install Carla Simulator, follow the instructions on the official website: https://carla.readthedocs.io/en/latest/start_quickstart/

### Images recording

To record images from the simulator, first launch the Carla server as expained on the official website :
https://carla.readthedocs.io/en/latest/start_quickstart/#running-carla

Then, launch the client with the example script provided :
```bash
python3 carla/PythonAPI/examples/manual_control.py
```

From there, the controls are displayed on the commad line. You can change the vehicle, the weather, etc. To start or stop recording images, press the "R" key.

## Usage

### Type and size of data

The BGR format or derivatives (like BGRDS) is and must be always used. It should be converted to RBG only when displaying the data.

Tensor sizes ((batch size), number of channels, height, width)
Numpy array size ((batch size), width, height, number of channels)

Data number of channels is a minimum of 3 and can be more.
Image number of channels is 3.

### Data naming convention

`hr` => high resolution data or image \
`lr` => low resolution data or image

`_data` => entry data that is getable using operation `[X]` where `X` is an index. \
`_img` => when the data is guaranteed to have 3 channels, we name the associated variable as stated.

In fact our datasets return all the low resolutions data and the associated high resolution data.

`_patch` => if the data is a subset of a greater size data. \
`_tensor` => if the data is a tensor \
`_np` => if the data is numpy array

+(s) if the data is a list 

**Examples**

`hr_data_patch_tensors` => a list of data patches represented as a tensor.

`lr_img_np` => an image presented as numpy array.


## Folders

### **resources**

All the datasets, used images and json data used for tests purpose should be placed here.

### **results**

All the weights, upscaled images and created information for tests purpose will be saved here.

Weights : 
```
weights-upscale-residual-lpips-v.2 : our best model, BGR model, uses rgb images

weights-upscale-residual2-lpips-v.4.1 : best BGRDS model, uses numpy array as images
```

### **src**

Contains all our "library" code.

### Current folder

#### *link.json*

A json file that contains our dataset links and their related information (reference channel position, channel position, method for each channel in order to create the lower size dataset).

#### *train_model.ipynb*

A Jupyter notebook to train a new model or to evaluate the performance of an existing one. There are four parts in this notebook. The first part is the context definition, where the user can customize its training. The second part (optional) is for the training of the overfit model, which can help the user know if, at least, the model can converge. The third part is the regular training of a model. And the fourth part aims to show different illustrations of super resolution and aims to compute several metrics. 

#### *test_model_on_image.ipynb*

A Jupyter notebook where the user can use a model and an image. This notebook works in two different ways. The "down resolution first", which will downscale the image by the upscale factor then try to reconstruct the original image using our model. We can use the original image as reference for metric computation or just for the user to judge. The "only super resolution" mode will only upscale the image by the upscale factor. 

#### *upscale_video.py*

This script serves to upscale a full mp4 video, it will only work with the "only super resolution" mode. 

#### *app.py*

A script that opens a local server using streamlit, in order to make the user choose an image that will be upscaled.

#### *MEGA_TEST.py*

Good name, isn't it ? In particular, this script takes as an argument the path to a json configuration file that will guide the tests. The configuration file will list the models to be tested, how to initialise them, the datasets to be tested, etc... The output is two results files. One file with the mean, variance, minimum and maximum of each score for the PSNR and SSIM metrics. And a second `_full` file with a list of all the scores for all the images in the datasets.

#### *run_scenarios.sh*

A bash script designed to run several test configurations.

#### *manipulate_result.py*

This script takes as its argument a method and the path to one or two result files. It allows the user to modify them in several ways.

#### *show_results.py*

This script takes as its argument a result path and many contexts argument. It will show the results using matplotlib.