from src.PytorchUtil import PytorchUtil as torchUtil

from skimage import metrics
import numpy as np
import torch
import math

class PatchImageTool:

    @staticmethod
    def get_patch_from_image_xy(image_torch, x, y, patch_size=32):
        start_x = patch_size * x
        start_y = patch_size * y

        end_x = start_x + patch_size
        end_y = start_y + patch_size

        if end_x > image_torch.shape[2]:
            end_x = image_torch.shape[2]
            start_x = end_x - patch_size

        if end_y > image_torch.shape[1]:
            end_y = image_torch.shape[1]
            start_y = end_y - patch_size

        return image_torch[:, start_y: end_y, start_x: end_x]

    @staticmethod
    def get_patch_from_image_index(image_torch, index, patch_size=32, w=None, h=None):
        if w is None:
            w = math.ceil(image_torch.shape[2] / patch_size)

        if h is None:
            h = math.ceil(image_torch.shape[1] / patch_size)

        x = index % w
        y = index // w

        return PatchImageTool.get_patch_from_image_xy(image_torch, x, y, patch_size=patch_size)


    @staticmethod
    def get_patchs_from_image(image_torch, patch_size=32, w=None, h=None):
        if w is None:
            w = math.ceil(image_torch.shape[2] / patch_size)

        if h is None:  
            h = math.ceil(image_torch.shape[1] / patch_size)    

        patches_res = torch.zeros((w * h, image_torch.shape[0], patch_size, patch_size), dtype=torch.float32)

        for i in range(h):
            for j in range(w):
                patches_res[i * w + j] = PatchImageTool.get_patch_from_image_xy(image_torch, j, i, patch_size=patch_size)

        return patches_res


    # Upscaling factor of this function is sf, so the predicted patch will be twice bigger
    # Use a batch size of 32 to compute the predicted patch
    @staticmethod
    def predict_image_from_image_patches(model, image_size, image_patches, device, patch_size=32, sf=2) -> torch.Tensor:
        predicted_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.float32)

        # Compute the number of patches in width and height
        num_patch_width = math.ceil(image_size[0] / sf / patch_size)
        num_patch_height = math.ceil(image_size[1] / sf / patch_size)
        num_patches = num_patch_width * num_patch_height
        
        batch_size = 1024

        computed_patch = []
        
        for i in range(0, num_patches, batch_size):
            # Number of patch and batch size might not be divisible
            # So we need to take the min of the two
            num_patch_to_compute = min(batch_size, num_patches - i)
            torch_patches = image_patches[i : i + num_patch_to_compute]

            with torch.no_grad():
                torch.cuda.empty_cache()
                batch = torch_patches.to(device, dtype=torch.float32)
                predicted_batch = model.net(batch)

                # Append each patch individually after transformed to numpy
                for j in range(len(predicted_batch)):
                    computed_patch.append(torchUtil.tensor_to_numpy(predicted_batch[j]))

        for i in range(num_patch_height):
            for j in range(num_patch_width):
                patch_index = i * num_patch_width + j
                patch = computed_patch[patch_index]
                
                image_x_start = j * patch_size * sf
                image_x_end = (j + 1) * patch_size * sf

                if image_x_end > image_size[0]:
                    image_x_end = image_size[0]
                    image_x_start = image_size[0] - patch_size * sf

                image_y_start = i * patch_size * sf
                image_y_end = (i + 1) * patch_size * sf

                if image_y_end > image_size[1]:
                    image_y_end = image_size[1]
                    image_y_start = image_size[1] - patch_size * sf

                predicted_image[image_y_start:image_y_end, image_x_start:image_x_end, :] = patch

        return torchUtil.numpy_to_tensor(predicted_image)

    @staticmethod
    def predict_image_from_dataset_patches(model, image_size, dataset, index, device, patch_size=32, sf=2) -> torch.Tensor:
                # Return a list of patch images
        low_res, _ = dataset.get_all_patch_for_image(index)

        return PatchImageTool.predict_image_from_image_patches(model, image_size, low_res, device, patch_size=patch_size, sf=sf)
    

    @staticmethod
    def predict_images_from_images_patches(model, image_size, number_image, images_patches, device, patch_size=32, sf=2, batch_size=2048) -> torch.Tensor:
        patches_on_one_image = len(images_patches) // number_image
        predicted_images = []

        computed_patch = []
        
        for i in range(0, len(images_patches), batch_size):
            # Number of patch and batch size might not be divisible
            # So we need to take the min of the two
            num_patch_to_compute = min(batch_size, len(images_patches) - i)
            torch_patches = images_patches[i : i + num_patch_to_compute]

            with torch.no_grad():
                torch.cuda.empty_cache()
                batch = torch_patches.to(device, dtype=torch.float32)
                predicted_batch = model.net(batch)

                # Append each patch individually after transformed to numpy
                for patch_index in range(len(predicted_batch)):
                    computed_patch.append(predicted_batch[patch_index].detach().to('cpu').numpy())

        num_patch_width = math.ceil(image_size[0] / sf / patch_size)

        # Reconstruction of the image

        for i in range(number_image):
            predicted_images.append(np.zeros((image_size[1], image_size[0], 3), dtype=np.float32))

            for j in range(patches_on_one_image):
                patch_index = i * patches_on_one_image + j
                patch = computed_patch[patch_index].transpose(1, 2, 0)
                
                image_x_start = j % num_patch_width * patch_size * sf
                image_x_end = (j % num_patch_width + 1) * patch_size * sf

                if image_x_end > image_size[0]:
                    image_x_end = image_size[0]
                    image_x_start = image_size[0] - patch_size * sf

                image_y_start = j // num_patch_width * patch_size * sf
                image_y_end = (j // num_patch_width + 1) * patch_size * sf

                if image_y_end > image_size[1]:
                    image_y_end = image_size[1]
                    image_y_start = image_size[1] - patch_size * sf

                predicted_images[i][image_y_start:image_y_end, image_x_start:image_x_end, :] = patch

        return [torchUtil.numpy_to_tensor(predicted_image) for predicted_image in predicted_images]
    

    @staticmethod
    # Predict multiple images at the same time in the case of batch size > patch per image
    def predict_images_from_dataset_patches(model, image_size, set, indices, device, patch_size=32, sf=2, batch_size=2048) -> torch.Tensor:
        patches_on_one_image = set.get_number_patch_per_image()
        patches_to_compute = torch.zeros((len(indices) * patches_on_one_image, 3, patch_size, patch_size), dtype=torch.float32)

        for i in range(len(indices)):
            image_patches, _ = set.get_all_patch_for_image(indices[i])

            for j in range(patches_on_one_image):
                patches_to_compute[i * patches_on_one_image + j] = image_patches[j]
        
        return PatchImageTool.predict_images_from_images_patches(
            model, 
            image_size, len(indices), patches_to_compute, 
            device, 
            patch_size=patch_size, sf=sf, batch_size=batch_size)


    @staticmethod
    def compute_metrics_dataset(model, dataset, sub_dataset_size, image_size, device, patch_size=32):
        psnr = np.zeros(sub_dataset_size)
        ssim = np.zeros(sub_dataset_size)

        # array of unique indices
        indices = np.random.choice(len(dataset), sub_dataset_size, replace=False)

        for i in range(sub_dataset_size):
            index_patch = indices[i]

            index_image = dataset.get_index_for_image(index_patch)
            
            _, high_res = dataset.get_full_image(index_image)
            pred_high_res = PatchImageTool.predict_image_from_dataset_patches(model, image_size, dataset, index_patch, device, patch_size=patch_size)

            high_res_np = torchUtil.tensor_to_numpy(high_res)
            pred_high_res_np = torchUtil.tensor_to_numpy(pred_high_res)

            psnr[i] = metrics.peak_signal_noise_ratio(high_res_np, pred_high_res_np)
            ssim[i] = metrics.structural_similarity(high_res_np, pred_high_res_np, win_size=7, data_range=1, multichannel=True, channel_axis=2)

            if i % (sub_dataset_size // 10) == 0:
                print("Current index", i, "PSNR", psnr[i], "SSIM", ssim[i])

        return psnr, ssim


    @staticmethod
    def compute_metrics_dataset_batched(model, image_size, dataset, sub_dataset_size, device, batch_size=2048):
        psnr = np.zeros(sub_dataset_size)
        ssim = np.zeros(sub_dataset_size)

        # array of unique indices
        number_patch_per_image = dataset.get_number_patch_per_image()
        number_image_per_gen = batch_size // number_patch_per_image
        indices_patch = np.zeros(sub_dataset_size, dtype=np.int32)
        indices_image = np.zeros(sub_dataset_size, dtype=np.int32)

        for i in range(sub_dataset_size):
            indices_patch[i] = number_patch_per_image * i

            indices_image[i] = dataset.get_index_for_image(indices_patch[i])

        # iterate on indices_patch, image by an overbatch of number_image_per_gen
        for i in range(0, sub_dataset_size, number_image_per_gen):
            sub_indices_patch = indices_patch[i:i+number_image_per_gen]
            sub_indices_image = indices_image[i:i+number_image_per_gen]

            predicted_images = PatchImageTool.predict_images_from_dataset_patches(model, image_size, dataset, sub_indices_patch, device, batch_size=batch_size)

            for j in range(len(predicted_images)):
                #print("Do step ", i + j, "out of", subpart_size)
                _, high_res = dataset.get_full_image(sub_indices_image[j])
                pred_high_res = predicted_images[j]

                high_res_np = torchUtil.tensor_to_numpy(high_res)
                pred_high_res_np = torchUtil.tensor_to_numpy(pred_high_res)

                psnr[i + j] = metrics.peak_signal_noise_ratio(high_res_np, pred_high_res_np)
                ssim[i + j] = metrics.structural_similarity(high_res_np, pred_high_res_np, win_size=7, data_range=1, multichannel=True, channel_axis=2)

        return psnr, ssim