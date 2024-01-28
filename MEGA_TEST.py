import os, sys
import json
import argparse
import platform
import copy

from src.utils.ImageTool import ImageTool
from src.dataset.PatchImageTool import PatchImageTool

from src.models.InitModel import InitModel

from src.utils.PytorchUtil import PytorchUtil as torchUtil
from src.dataset.ImageDataset import ImageDataset
from src.dataset.ImageDatasetPatch import ImageDatasetPatch

import torch
import torchvision.transforms as transforms
import torchvision.transforms as transforms
import torch.utils.data as data

import numpy as np

import skimage.metrics as metrics

# Create an arg parser
# Can take a path as argument (optional)
# Can take an option -r, wich means that a new config file will be created and erased if one already exists
# Can take an option -h to display help
# Can take an option -nv to verbose the program
# Can take an option -nf to not save the result in a file
def create_arg_parse():
    parser = argparse.ArgumentParser(description='Test the MEGA project')
    parser.add_argument('-p', '--path', type=str, help='path to the config file')
    parser.add_argument('-r', '--reset', action='store_true', help='reset the config file')
    parser.add_argument('-nv', '--no-verbose', action='store_true', help='not verbose the program')
    parser.add_argument('-nf', '--no-file', action='store_true', help='not save the result in a file')

    return parser

# check if resources folder exists
if not os.path.exists('resources'):
    os.makedirs('resources')


def create_test_config() -> dict:
    return {
        "resultNameFile" : "result",
        "models" : [
            {
                "name" : "upscale",
                "type" : "neuralNetwork",
                "weights" : "superresol-upscale",
                "hyperparameters" : {
                    "learningRate" : 0.001,
                    "channels" : ["b", "g", "r"],
                    "channel_interpolations" : ["bicubic", "bicubic","bicubic"]
                },
            },
            {
                "name" : "bilinear",
                "type" : "alternative",
            },
            {
                "name" : "bicubic",
                "type" : "alternative",
            },
            {
                "name" : "nearest",
                "type" : "alternative",
            }
        ],
        "dataset" : ["train", "test"],
        "upscaleFactors" : [
             2, 4, 8
        ],
        "upscalingMethods" : [
            {
                "method" : "patch",
                "patchSize" : 256,
                "batchSize" : 2048
            },
            { 
                "method" : "image",
                "batchSize" : 3
            }
        ]
    }

def get_device():
    device = None
    if platform.system() == 'Windows':  # Check if the OS is Windows
        import torch_directml  # Import torch_directml only on Windows
        device = torch_directml.device()

    force_cpu = False

    if not device:
        if torch.cuda.is_available() and not force_cpu:
            device = torch.device('cuda')
            torch.cuda.empty_cache()
        else:
            device = torch.device('cpu')

    return device

def get_dataset(dataset_name: str, upscale_factor_list: list, channels: list, patch_size: int=None):
    common_transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    main_resolution = (1920, 1080)
    main_res_str = "{}x{}".format(main_resolution[0], main_resolution[1])

    if patch_size is None:
        return ImageDataset(dataset_name, main_res_str, upscale_factor_list, channels, transforms=common_transform, 
                            download=True, verbose=True)
    
    return ImageDatasetPatch(dataset_name, main_res_str, upscale_factor_list, channels, transforms=common_transform, 
                             download=True, patch_size=patch_size, verbose=True)

# Return the array of the metrics
def compute_metrics(dataloader, method, model, upscale_factor_list, device):
    dataset = dataloader.dataset

    if method["method"] == "patch":
        upscale_factor = model.net.get_upscale_mode()
        size = int(len(dataset) / dataset.get_number_patch_per_image(upscale_factor=upscale_factor))

        return PatchImageTool.compute_metrics_dataset_batched(
                        model, 
                        dataset.hr_data_size, 
                        dataset, size, 
                        device, method["batchSize"], verbose=True)
    elif method["method"] == "image":
        upscale_index = upscale_factor_list.index(model.net.get_upscale_mode())
        return ImageTool.compute_metrics_dataset(model, dataloader, upscale_index, device, verbose=True)
    else:
        raise Exception("The method name is not correct")
    
def compute_metrics_alternative_method(dataloader, method, altertive_method, upscale_index, device, verbose=False):
    dataset = dataloader.dataset

    psnr = np.zeros(len(dataset))
    ssim = np.zeros(len(dataset))

    image_size = (dataset[0][1].shape[1], dataset[0][1].shape[2])

    if "bilinear" in altertive_method:
        resize_function = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
    elif "bicubic" in altertive_method:
        resize_function = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    elif "nearest" in altertive_method:
        resize_function = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST, antialias=True)

    batch_size = method["batchSize"]

    # get data and idx        
    for i, (lr_data_tensors, hr_img_tensor) in enumerate(dataloader):
        with torch.no_grad():
            lr_data_tensor = lr_data_tensors[upscale_index].to(device)

            index = i * batch_size
            end = min(index + batch_size, len(dataset))

            resized_img_tensors = resize_function(lr_data_tensor)

            for j in range(0, end - index):
                resized_img_np = torchUtil.tensor_to_numpy(resized_img_tensors[j])
                hr_img_np = torchUtil.tensor_to_numpy(hr_img_tensor[j])
                psnr[j + index] = metrics.peak_signal_noise_ratio(hr_img_np, resized_img_np)

                ssim[j + index] = metrics.structural_similarity(
                    hr_img_np, resized_img_np, win_size=7, 
                    data_range=1, multichannel=True, channel_axis=2)

            # if verbose and ervery 1 %
            if verbose and (index) % (len(dataset) // 20) == 0:
                print("{}%".format(index / len(dataset) * 100))

    return psnr, ssim


def save_results(result, result_full, datasetName, upscaleFactor, method, model, metrics):
    entry = {
        "dataset" : datasetName,
        "upscaleFactor" : upscaleFactor,
        "method" : method,
        "model" : model
    }

    for metrics_name, metrics_value in metrics.items():
        entry[metrics_name] = {
            "mean" : np.mean(metrics_value),
            "var" : np.var(metrics_value),
            "min" : np.min(metrics_value),
            "max" : np.max(metrics_value),
        }

    entry_full = copy.deepcopy(entry)

    for metrics_name, metrics_value in metrics.items():
        entry_full[metrics_name]["values"] = metrics_value.tolist()

    result["entries"].append(entry)
    result_full["entries"].append(entry_full)

    with open(result_file_path + ".json", 'w') as f:
        json.dump(result, f, indent=4)

    with open(result_file_path + "_full.json", 'w') as f:
        json.dump(result_full, f, indent=4)


if __name__ == "__main__":
    resources_path = "resources/"
    default_file = resources_path + "default_config.json"

    results_path = "results/"

    # Create the arg parser
    parser = create_arg_parse()
    prog_args = parser.parse_args()

    def printf(*args, **kwargs):
        if not prog_args.no_verbose:
            # call the print function from the builtins
            print(*args, **kwargs)

    # Check if the path has been given as argument
    print(prog_args.path)
    if prog_args.path:
        config_path = resources_path + prog_args.path
        printf("* Using config file at \n\t- {}\n".format(config_path))
    else:
        config_path = default_file
        printf("* Using default config file at\n\t- {}\n".format(config_path))
    
    # Check if should erase the config file
    if prog_args.reset:
        printf("* Erasing config file in order to create new one \n\t- {}\n".format(config_path))
        if os.path.exists(config_path):
            os.remove(config_path)

    if not os.path.exists(config_path) or prog_args.reset:
        with open(config_path, 'w') as f:
            printf("* Creating new config file at \n\t- {}\n".format(config_path))
            json.dump(create_test_config(), f, indent=4)
    
    config = json.load(open(config_path))

    result = {"entries" : []}
    result_full = {"entries" : []}

    if prog_args.no_file:
        printf("* All the experiment will not written in the console only")
    else:
        # All the experiment result will be save 
        printf("* All the experiment will be written and saved in the file \n\t- {}".format(config["resultNameFile"]))

        # Create or reset the file
        result_file_path = results_path + config["resultNameFile"]

        if os.path.exists(result_file_path):
            os.remove(result_file_path)

    torch_device = get_device()

    with torch.no_grad():
        for model in config["models"]:
            if model["type"] != "alternative":
                try:
                    nn_model = InitModel.create_model_static(
                            model["name"], model["weights"], model["hyperparameters"], 
                            2, torch_device)
                except Exception as e:
                    raise Exception(e)


        printf("* Using device : {}".format(torch_device))
        printf("")

        for dataset_name in config["dataset"]:
            printf("* * Using dataset : {}".format(dataset_name))

            for method in config["upscalingMethods"]:
                printf("** * Using method : {}".format(method["method"]))

                patch_size = method["patchSize"] if method["method"].lower() == "patch" else None

                for upscale_factor in config["upscaleFactors"]:
                    printf("*** * Using upscale factor : {}".format(upscale_factor))

                    for model in config["models"]:
                        psnr, ssim = None, None

                        if model["type"] == "alternative":
                            # not usefull if not image
                            if method["method"].lower() == "image":
                                dataset = get_dataset(dataset_name, config["upscaleFactors"], ["b", "g", "r"], patch_size)
                                # create an enumarate to get batches, use torch.utils.data.DataLoader
                                dataloader = data.DataLoader(dataset, batch_size=method["batchSize"], shuffle=False)

                                batch_size = dataloader.batch_size

                                printf("**** * Using alternative method : {}".format(model["name"]))

                                psnr, ssim = compute_metrics_alternative_method(dataloader,
                                                                                method, model["name"], 
                                                                                config["upscaleFactors"].index(upscale_factor),
                                                                                torch_device, verbose=True)
                        else:
                            print ("**** * Using neural network model : {}".format(model["name"]))
                            dataset = get_dataset(dataset_name, config["upscaleFactors"],
                                                   model["hyperparameters"]["channel_positions"] , patch_size)
                            
                            # create an enumarate to get batches, use torch.utils.data.DataLoader
                            dataloader = data.DataLoader(dataset, batch_size=method["batchSize"], shuffle=False)

                            batch_size = dataloader.batch_size

                            nn_model = InitModel.create_model_static(model["name"], 
                                                                     model["weights"], model["hyperparameters"], 
                                                                     upscale_factor, torch_device)
                            
                            psnr, ssim = compute_metrics(dataloader, method, nn_model, config["upscaleFactors"], torch_device)

                        if psnr is not None and ssim is not None:
                            # Show mean, var, min, max
                            printf("-> Mean PSNR : {}, var : {}, min : {}, max : {}".format(
                                np.mean(psnr), np.var(psnr), np.min(psnr), np.max(psnr)))
                            printf("-> Mean SSIM : {}, var : {}, min : {}, max : {}".format(np.mean(
                                ssim), np.var(ssim), np.min(ssim), np.max(ssim)))

                            # Add the result to result dict
                            if not prog_args.no_file:
                                save_results(result, result_full, dataset_name, upscale_factor, method, model, {
                                    "psnr" : psnr,
                                    "ssim" : ssim
                                })
