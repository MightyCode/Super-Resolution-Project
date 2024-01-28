from src.utils.PytorchUtil import PytorchUtil as torchUtil

from skimage import metrics
import numpy as np
import torch
import math

class PatchImageTool:

    """
    Get a patch from an image from a x and y position
    """
    @staticmethod
    def get_patch_from_image_xy(img_tensor: torch.Tensor, 
                                x: int, y: int, 
                                patch_size:int=32) -> torch.Tensor:
        
        start_x = patch_size * x
        start_y = patch_size * y

        end_x = start_x + patch_size
        end_y = start_y + patch_size

        if end_x > img_tensor.shape[2]:
            end_x = img_tensor.shape[2]
            start_x = end_x - patch_size

        if end_y > img_tensor.shape[1]:
            end_y = img_tensor.shape[1]
            start_y = end_y - patch_size

        return img_tensor[:, start_y: end_y, start_x: end_x]

    """
    Get a patch from an image
    """
    @staticmethod
    def get_patch_from_image_index(img_tensor: torch.Tensor, 
                                   index: int, 
                                   patch_size: int=32, 
                                   w: int=None, h: int=None) -> torch.Tensor:
        
        if w is None:
            w = math.ceil(img_tensor.shape[2] / patch_size)

        if h is None:
            h = math.ceil(img_tensor.shape[1] / patch_size)

        x = index % w
        y = index // w

        return PatchImageTool.get_patch_from_image_xy(img_tensor, x, y, patch_size=patch_size)

    """
    Get all the patches from an image
    """
    @staticmethod
    def get_patchs_from_image(img_tensor: torch.Tensor, 
                              patch_size: int=32, 
                              w: int=None, h: int=None) -> list:
        
        if w is None:
            w = math.ceil(img_tensor.shape[2] / patch_size)

        if h is None:  
            h = math.ceil(img_tensor.shape[1] / patch_size)    

        patch_tensors = torch.zeros((w * h, img_tensor.shape[0], patch_size, patch_size), dtype=torch.float32)

        for i in range(h):
            for j in range(w):
                patch_tensors[i * w + j] = PatchImageTool.get_patch_from_image_xy(img_tensor, j, i, patch_size=patch_size)

        return patch_tensors


    """
    Upscaling factor of this function is upscale_factor, so the predicted patch will be x bigger
    Use a batch size of 32 to compute the predicted patch
    """
    @staticmethod
    def predict_image_from_image_patches(model, 
                                         lr_img_size, lr_data_patch_tensors: list, 
                                         device, 
                                         patch_size: int=32, upscale_factor: int=2) -> torch.Tensor:
        pred_img_np = np.zeros((lr_img_size[1], lr_img_size[0], 3), dtype=np.float32)

        # Compute the number of patches in width and height
        num_patch_width = math.ceil(lr_img_size[0] / upscale_factor / patch_size )
        num_patch_height = math.ceil(lr_img_size[1] / upscale_factor / patch_size )
        num_patches = num_patch_width * num_patch_height

        upscale_patch_size = int(patch_size * upscale_factor)
        
        batch_size = 1024

        computed_patch_np = []
        
        model.net.set_upscale_mode(upscale_factor)

        for i in range(0, num_patches, batch_size):
            # Number of patch and batch size might not be divisible
            # So we need to take the min of the two
            num_patch_to_compute = min(batch_size, num_patches - i)
            lr_data_patch_tensors = lr_data_patch_tensors[i : i + num_patch_to_compute]

            with torch.no_grad():
                torch.cuda.empty_cache()
                batch = lr_data_patch_tensors.to(device, dtype=torch.float32)
                pred_img_patch_tensor = model.net(batch)

                # Append each patch individually after transformed to numpy
                for j in range(len(pred_img_patch_tensor)):
                    computed_patch_np.append(torchUtil.tensor_to_numpy(pred_img_patch_tensor[j]))

        for i in range(num_patch_height):
            for j in range(num_patch_width):
                patch_index = i * num_patch_width + j
                patch = computed_patch_np[patch_index]
                
                image_x_start = j * upscale_patch_size
                image_x_end = (j + 1) * upscale_patch_size

                if image_x_end > lr_img_size[0]:
                    image_x_end = lr_img_size[0]
                    image_x_start = lr_img_size[0] - upscale_patch_size

                image_y_start = i * upscale_patch_size
                image_y_end = (i + 1) * upscale_patch_size

                if image_y_end > lr_img_size[1]:
                    image_y_end = lr_img_size[1]
                    image_y_start = lr_img_size[1] - upscale_patch_size

                pred_img_np[image_y_start:image_y_end, image_x_start:image_x_end, :] = patch

        return torchUtil.numpy_to_tensor(pred_img_np)

    """
    Predict single image from its patches
    """
    @staticmethod
    def predict_image_from_dataset_patches(model, img_size, dataset, index, device) -> torch.Tensor:
        upscale = model.net.get_upscale_mode()
        # Return a list of patch images
        lr_data_patch_tensors, _ = dataset.get_all_patch_for_image(index, upscale)

        return PatchImageTool.predict_image_from_image_patches(
            model, 
            img_size, lr_data_patch_tensors, 
            device, 
            patch_size=dataset.get_patch_size(upscale_factor=upscale), upscale_factor=upscale)


    """
    Predict multiple image from their patches
    """
    @staticmethod
    def predict_images_from_images_patches(
                        model, 
                        lr_img_size, number_image, lr_data_patch_tensors: list, 
                        device, 
                        patch_size=32, upscale_factor=2, batch_size=2048) -> torch.Tensor:
        
        with torch.no_grad():
            torch.cuda.empty_cache()

            upscale_patch_size = int(patch_size * upscale_factor)
            number_patch_on_image = len(lr_data_patch_tensors) // number_image

            computed_img_patch_nps = []
            
            for i in range(0, len(lr_data_patch_tensors), batch_size):
                # Number of patch and batch size might not be divisible
                # So we need to take the min of the two
                num_patch_to_compute = min(batch_size, len(lr_data_patch_tensors) - i)
                batch: list = lr_data_patch_tensors[i : i + num_patch_to_compute]
                batch = batch.to(device, dtype=torch.float32)
                pred_img_patch_tensors = model.net(batch)

                # Append each patch individually after transformed to numpy
                for patch_index in range(len(pred_img_patch_tensors)):
                    computed_img_patch_nps.append(pred_img_patch_tensors[patch_index].detach().to('cpu').numpy())

            num_patch_width = math.ceil(lr_img_size[0] / upscale_factor / patch_size)

            # Reconstruction of the images
            pred_img_nps = []

            for i in range(number_image):
                pred_img_nps.append(np.zeros((lr_img_size[1], lr_img_size[0], 3), dtype=np.float32))

                for j in range(number_patch_on_image):
                    patch_index = i * number_patch_on_image + j
                    computed_img_patch = computed_img_patch_nps[patch_index].transpose(1, 2, 0)
                    
                    image_x_start = j % num_patch_width * upscale_patch_size
                    image_x_end = (j % num_patch_width + 1) * upscale_patch_size

                    if image_x_end > lr_img_size[0]:
                        image_x_end = lr_img_size[0]
                        image_x_start = lr_img_size[0] - upscale_patch_size

                    image_y_start = j // num_patch_width * upscale_patch_size
                    image_y_end = (j // num_patch_width + 1) * upscale_patch_size

                    if image_y_end > lr_img_size[1]:
                        image_y_end = lr_img_size[1]
                        image_y_start = lr_img_size[1] - upscale_patch_size

                    pred_img_nps[i][image_y_start:image_y_end, image_x_start:image_x_end, :] = computed_img_patch

            return [torchUtil.numpy_to_tensor(predicted_image) for predicted_image in pred_img_nps]
    
    """
    Predict multiple images from their patches
    """
    @staticmethod
    # Predict multiple images at the same time in the case of batch size > patch per image
    def predict_images_from_dataset_patches(model, lr_img_size, 
                                            dataset, indices, 
                                            device, 
                                            batch_size: int=2048) -> torch.Tensor:
        upscale_factor = model.net.get_upscale_mode()
        lr_patch_size = dataset.get_patch_size(upscale_factor=upscale_factor)

        patches_on_one_image = dataset.get_number_patch_per_image(upscale_factor=upscale_factor)
        lr_data_patch_tensors = torch.zeros((len(indices) * patches_on_one_image, model.net.get_num_channel(), 
                                             lr_patch_size, lr_patch_size), dtype=torch.float32)

        for i in range(len(indices)):
            lr_subdata_patch_tensors, _ = dataset.get_all_patch_for_image(indices[i], upscale_factor=upscale_factor)

            for j in range(patches_on_one_image):
                lr_data_patch_tensors[i * patches_on_one_image + j] = lr_subdata_patch_tensors[j]

        return PatchImageTool.predict_images_from_images_patches(
            model, 
            lr_img_size, len(indices), lr_data_patch_tensors, 
            device, 
            patch_size=lr_patch_size, upscale_factor=upscale_factor, batch_size=batch_size)

    """
    Predict multiple images from their patches
    """
    @staticmethod
    def compute_metrics_dataset(model,
                                 dataset, sub_dataset_size, 
                                 lr_img_size, 
                                 device, 
                                 verbose=False):
        
        psnr = np.zeros(sub_dataset_size)
        ssim = np.zeros(sub_dataset_size)

        # array of unique indices
        indices = np.random.choice(len(dataset), sub_dataset_size, replace=False)

        for i in range(sub_dataset_size):
            index_patch = indices[i]
            index_image = dataset.get_index_for_image(index_patch)
            
            with torch.no_grad():
                _, hr_img_tensor = dataset.get_full_image(index_image)

                pred_img_tensor = PatchImageTool.predict_image_from_dataset_patches(
                            model, 
                            lr_img_size, dataset, index_patch, device)

                hr_img_np = torchUtil.tensor_to_numpy(hr_img_tensor)
                pred_np = torchUtil.tensor_to_numpy(pred_img_tensor)

                psnr[i] = metrics.peak_signal_noise_ratio(hr_img_np, pred_np)
                ssim[i] = metrics.structural_similarity(hr_img_np, pred_np, win_size=7, 
                                                        data_range=1, multichannel=True, channel_axis=2)

                if verbose and i % (sub_dataset_size // 100) == 0:
                    print("Current index", i, "PSNR", psnr[i], "SSIM", ssim[i])

        return psnr, ssim

    """
    Predict multiple images from their patches
    """
    @staticmethod
    def compute_metrics_dataset_batched(model, lr_img_size, 
                                        dataset, sub_dataset_size, 
                                        device, 
                                        batch_size :int=2048, verbose=False):
        with torch.no_grad():
                
            psnr = np.zeros(sub_dataset_size)
            ssim = np.zeros(sub_dataset_size)

            upscale_factor = model.net.get_upscale_mode()

            # array of unique indices
            number_patch_per_image = dataset.get_number_patch_per_image(upscale_factor=upscale_factor)
            number_image_per_gen = math.ceil(batch_size / number_patch_per_image)

            indices_patch = np.zeros(sub_dataset_size, dtype=np.int32)
            indices_image = np.zeros(sub_dataset_size, dtype=np.int32)

            for i in range(sub_dataset_size):
                indices_patch[i] = number_patch_per_image * i

                indices_image[i] = dataset.get_index_for_image(indices_patch[i])

            # iterate on indices_patch, image by an overbatch of number_image_per_gen
            for i in range(0, sub_dataset_size, number_image_per_gen):
                sub_indices_patch = indices_patch[i:i+number_image_per_gen]
                sub_indices_image = indices_image[i:i+number_image_per_gen]

                pred_img_tensors: list = PatchImageTool.predict_images_from_dataset_patches(model, lr_img_size, dataset, sub_indices_patch, 
                                                                                    device, batch_size=batch_size)
                for j in range(len(pred_img_tensors)):
                    #print("Do step ", i + j, "out of", subpart_size)
                    _, hr_img_tensor = dataset.get_full_image(sub_indices_image[j])
                    pred_img_tensor = pred_img_tensors[j]

                    high_res_np = torchUtil.tensor_to_numpy(hr_img_tensor)
                    pred_hr_img_np = torchUtil.tensor_to_numpy(pred_img_tensor)

                    psnr[i + j] = metrics.peak_signal_noise_ratio(high_res_np, pred_hr_img_np)
                    ssim[i + j] = metrics.structural_similarity(high_res_np, pred_hr_img_np, win_size=7, 
                                                                data_range=1, multichannel=True, channel_axis=2)

                    if verbose and (i + j) % (sub_dataset_size // 20) == 0:
                        print((i + j) / sub_dataset_size * 100, "% (", (i + j) , ") -> PSNR", psnr[i + j], "SSIM", ssim[i + j])

            return psnr, ssim