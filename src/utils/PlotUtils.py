from .PytorchUtil import PytorchUtil as torchUtil

import matplotlib.pyplot as plt
import numpy as np
import torch


class PlotUtils:
    @staticmethod
    def show_high_low_res_images(low_res_images, high_res_images, upscales=None, width=15, name=""):
        assert len(low_res_images) == len(high_res_images)

        num_images = len(low_res_images)

        _, ax = plt.subplots(num_images, 2, figsize=(width, 5 * num_images))

        if num_images == 1:
            low_res_image = torchUtil.tensor_to_image(low_res_images[0])
            high_res_image = torchUtil.tensor_to_image(high_res_images[0])
            
            ax[0].imshow(low_res_image)
            ax[0].set_title(name + " Low resolution" + ("" if upscales is None else " (x" + \
                                                         str(upscales[0]) + ")" + str(low_res_image.shape)))
            ax[1].imshow(high_res_image)
            ax[1].set_title(name + " High resolution (" + str(high_res_image.shape) + ")")

            plt.show()
            
            return

        for i in range(num_images):
            low_res_image = torchUtil.tensor_to_image(low_res_images[i])
            high_res_image = torchUtil.tensor_to_image(high_res_images[i])
            
            ax[i, 0].imshow(low_res_image)
            ax[i, 0].set_title(name + " Low resolution (" + str(i) + ")" + \
                                "" if upscales is None else " (x" + str(upscales[i]) + ")" + str(low_res_image.shape))
            ax[i, 1].imshow(high_res_image)
            ax[i, 1].set_title(name + " High resolution (" + str(i) + ")" + str(high_res_image.shape))

        plt.show()

    @staticmethod
    def show_dataset_example(dataset, num_images=1, indices=None, width=15):
        low_res_images = []
        high_res_images = []
        upscale_factors = []

        num_images = max(num_images, len(indices) if indices else 0)

        for i in range(num_images):
            if indices:
                low_res_patches, high_res = dataset[indices[i]]

                index_low_res_patch = np.random.randint(len(low_res_patches))
                
                low_res_images.append(low_res_patches[index_low_res_patch])
                high_res_images.append(high_res)
                # torch to int
                upscale_factors.append(dataset.get_upscale_factor(index_low_res_patch))
            else:
                index = np.random.randint(len(dataset))
                low_res_patches, high_res = dataset[index]

                index_low_res_patch = np.random.randint(len(low_res_patches))
                
                low_res_images.append(low_res_patches[index_low_res_patch])
                high_res_images.append(high_res)
                upscale_factors.append(dataset.get_upscale_factor(index_low_res_patch))
        
        PlotUtils.show_high_low_res_images(low_res_images, high_res_images, upscales=upscale_factors,
                                               width=width, name=dataset.name())


    # Plot for the the predicted image, low resolution image and high resolution image in first row
    # plot Plot the loss, psnr and ssim curves in the second row
    @staticmethod
    def plot_images_and_metrics(exp, axes, dataset, index, device): 
        axes[0][0].clear()
        axes[0][1].clear()
        axes[0][2].clear()
        axes[1][0].clear()
        axes[1][1].clear()
        axes[1][2].clear()

        ##Only to use when perform_validation_during_training == True
        low_res_patches, high_res = dataset[index]

        chosen_upscale = 0
        # Take first upscale to show performence
        low_res = low_res_patches[chosen_upscale]

        with torch.no_grad():
            predicted_res = exp.net(low_res.to(device))[0]

            low_res_image = torchUtil.tensor_to_image(low_res)
            high_res_image = torchUtil.tensor_to_image(high_res)
            predicted_res_image = torchUtil.tensor_to_image(predicted_res)

        axes[0][0].set_title(f'Low res (x{dataset.get_upscale_factor(chosen_upscale)}): {low_res_image.shape}')
        axes[0][1].set_title(f'High res: {high_res_image.shape}')
        axes[0][2].set_title(f'Predicted res: {predicted_res_image.shape}')

        axes[0][0].imshow(low_res_image)
        axes[0][1].imshow(high_res_image)
        axes[0][2].imshow(predicted_res_image)

        axes[1][0].plot([exp.history[k][0]['loss'] for k in range(exp.epoch)], label="Train loss")
        axes[1][1].plot([exp.history[k][0]['psnr'] for k in range(exp.epoch)], label="Train psnr")
        axes[1][2].plot([exp.history[k][0]['ssim'] for k in range(exp.epoch)], label="Train ssim")

        axes[1][0].plot([exp.history[k][1]['loss'] for k in range(exp.epoch)], label="Eval loss")
        axes[1][1].plot([exp.history[k][1]['psnr'] for k in range(exp.epoch)], label="Eval psnr")
        axes[1][2].plot([exp.history[k][1]['ssim'] for k in range(exp.epoch)], label="Eval ssim")

        axes[1][0].legend()
        axes[1][0].set_xlabel("Epoch")
        axes[1][0].set_ylabel("Loss")        
        axes[1][1].legend()
        axes[1][1].set_xlabel("Epoch")
        axes[1][1].set_ylabel("PSNR") 
        axes[1][2].legend()
        axes[1][2].set_xlabel("Epoch")
        axes[1][2].set_ylabel("SSIM")

    # Show three images for a set and predict it
    @staticmethod
    def plot_images_from_model(model, dataset, device, num_images=1, indices=None):
        num_images = max(num_images, len(indices) if indices else 0)

        _, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

        if num_images == 1:
            axes[0].clear()
            axes[1].clear()
            axes[2].clear()

            with torch.no_grad():
                if indices:
                    low_res_patches, high_res = dataset[indices[0]]
                    print("Chosen index", indices[0])
                else:
                    index = np.random.randint(len(dataset))
                    low_res_patches, high_res = dataset[index]
                    print("Chosen index", index)

                chosen_upscale = 0
                low_res = low_res_patches[chosen_upscale]
                model.net.set_upscale_mode(chosen_upscale)

                predicted_res = model.net(low_res.to(device))[0]

                low_res_image = torchUtil.tensor_to_image(low_res)
                high_res_image = torchUtil.tensor_to_image(high_res)
                predicted_res_image = torchUtil.tensor_to_image(predicted_res)

            axes[0].set_title(f'Low res (x{dataset.get_upscale_factor(low_res_patches)}): {low_res_image.shape}')
            axes[1].set_title(f'High res: {high_res_image.shape}')
            axes[2].set_title(f'Predicted res: {predicted_res_image.shape}')

            axes[0].imshow(low_res_image)
            axes[1].imshow(high_res_image)
            axes[2].imshow(predicted_res_image)
        
            plt.show()

            return

        for i in range(num_images):
            with torch.no_grad():
                if indices:
                    low_res_patches, high_res = dataset[indices[i]]
                    print("Chosen index", indices[i])
                else:
                    index = np.random.randint(len(dataset))
                    low_res_patches, high_res = dataset[index]
                    print("Chosen index", index)

                chosen_upscale = 0
                low_res = low_res_patches[chosen_upscale]
                upscale = dataset.get_upscale_factor(chosen_upscale)

                model.net.set_upscale_mode(upscale)

                predicted_res = model.net(low_res.to(device))[0]

                low_res_image = torchUtil.tensor_to_image(low_res)
                high_res_image = torchUtil.tensor_to_image(high_res)
                predicted_res_image = torchUtil.numpy_to_image(torchUtil.tensor_to_numpy(predicted_res))

            axes[i, 0].set_title(f'Low res (x{upscale}): {low_res_image.shape}')
            axes[i, 1].set_title(f'High res: {high_res_image.shape}')
            axes[i, 2].set_title(f'Predicted res: {predicted_res_image.shape}')

            axes[i, 0].imshow(low_res_image)
            axes[i, 1].imshow(high_res_image)
            axes[i, 2].imshow(predicted_res_image)
        
        plt.show()

    @staticmethod
    def plot_predicted_and_bicubic(model, dataset, device, num_images=1, indices=None):
        num_images = max(num_images, len(indices) if indices else 0)

        _, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

        if num_images == 1:
            axes[0].clear()
            axes[1].clear()
            axes[2].clear()

            with torch.no_grad():
                if indices:
                    low_res_patches, _ = dataset[indices[0]]
                    print("Chosen index", indices[0])
                else:
                    index = np.random.randint(len(dataset))
                    low_res_patches, _ = dataset[index]
                    print("Chosen index", index)

                chosen_upscale = 0
                low_res = low_res_patches[chosen_upscale]
                upscale = dataset.get_upscale_factor(chosen_upscale)
                model.net.set_upscale_mode(upscale)
           
                predicted_torch = model.net(low_res.to(device))[0]

                predicted_res = torchUtil.tensor_to_numpy(predicted_torch)
                            
                bicubic_image = torchUtil.resize_tensor_to_numpy(low_res, (predicted_res.shape[0], predicted_res.shape[1]))
                subtraction_image = torchUtil.norm_numpy_image(predicted_res - bicubic_image)

            print(subtraction_image.mean(), subtraction_image.std())

            axes[0].set_title(f'Predicted res (x{upscale}): {predicted_res.shape}')
            axes[1].set_title(f'Bicubic res: {bicubic_image.shape}')
            axes[2].set_title(f'Substraction res: {subtraction_image.shape}')

            axes[0].imshow(torchUtil.numpy_to_image(predicted_res))
            axes[1].imshow(torchUtil.numpy_to_image(bicubic_image))
            axes[2].imshow(torchUtil.numpy_to_image(subtraction_image), vmin=subtraction_image.min(), vmax=subtraction_image.max())  
        
            plt.show()

            return


        for i in range(num_images):
            with torch.no_grad():
                if indices:
                    low_res_patches, _ = dataset[indices[i]]
                    print("Chosen index", indices[i])
                else:
                    index = np.random.randint(len(dataset))
                    low_res_patches, _ = dataset[index]
                    print("Chosen index", index)

                chosen_upscale = 0
                low_res = low_res_patches[chosen_upscale]
                upscale = dataset.get_upscale_factor(chosen_upscale)
                model.net.set_upscale_mode(upscale)
            
                predicted_torch = model.net(low_res.to(device))[0]

                predicted_res = torchUtil.tensor_to_numpy(predicted_torch)
                            
                bicubic_image = torchUtil.resize_tensor_to_numpy(low_res, (predicted_res.shape[0], predicted_res.shape[1]))
                subtraction_image = torchUtil.norm_numpy_image(predicted_res - bicubic_image)

            print(subtraction_image.mean(), subtraction_image.std())

            axes[i, 0].set_title(f'Predicted res (x{upscale}): {predicted_res.shape}')
            axes[i, 1].set_title(f'Bicubic res: {bicubic_image.shape}')
            axes[i, 2].set_title(f'Substraction res: {subtraction_image.shape}')

            axes[i, 0].imshow(torchUtil.numpy_to_image(predicted_res), vmin=0, vmax=1)
            axes[i, 1].imshow(torchUtil.numpy_to_image(bicubic_image), vmin=0, vmax=1)
            axes[i, 2].imshow(torchUtil.numpy_to_image(subtraction_image), vmin=subtraction_image.min(), vmax=subtraction_image.max())  
        
        plt.show()

    
    @staticmethod
    def show_low_high_predicted(low_image_tensors, high_image_tensor, predicted_image_tensor, name=""):
        assert len(low_image_tensors) == len(high_image_tensor) == len(predicted_image_tensor)

        _, axes = plt.subplots(len(low_image_tensors), 3, figsize=(30, 5 * len(low_image_tensors)))

        if len(low_image_tensors) == 1:
            with torch.no_grad():
                low_image = torchUtil.tensor_to_image(low_image_tensors[0])
                high_image = torchUtil.tensor_to_image(high_image_tensor[0])
                predicted_image = torchUtil.tensor_to_image(predicted_image_tensor[0])
            
            axes[0].set_title(f'{name} Low res: {low_image.shape}')
            axes[1].set_title(f'{name} High res: {high_image.shape}')
            axes[2].set_title(f'{name} Predicted res: {predicted_image.shape}')

            axes[0].imshow(low_image)
            axes[1].imshow(high_image)
            axes[2].imshow(predicted_image)

            plt.show()
            
            return

        for i in range(len(low_image_tensors)):
            with torch.no_grad():
                low_image = torchUtil.tensor_to_image(low_image_tensors[i])
                high_image = torchUtil.tensor_to_image(high_image_tensor[i])
                predicted_image = torchUtil.tensor_to_image(predicted_image_tensor[i])
            
            axes[i, 0].set_title(f'{name} Low res: {low_image.shape}')
            axes[i, 1].set_title(f'{name} High res: {high_image.shape}')
            axes[i, 2].set_title(f'{name} Predicted res: {predicted_image.shape}')

            axes[i, 0].imshow(low_image)
            axes[i, 1].imshow(high_image)
            axes[i, 2].imshow(predicted_image)

        plt.show()

    @staticmethod
    def show_predicted_interpolated_subtraction( 
            predicted_images_numpy, 
            interpolated_images, 
            substraction_images_numpy, 
            name=""):
        
        assert len(predicted_images_numpy) == len(interpolated_images) == len(substraction_images_numpy)

        _, axes = plt.subplots(len(predicted_images_numpy), 3, figsize=(30, 5 * len(predicted_images_numpy)))

        if len(predicted_images_numpy) == 1:
            predicted_image = torchUtil.numpy_to_image(predicted_images_numpy[0])
            interpolated_image = torchUtil.numpy_to_image(interpolated_images[0])
            subtraction_image = torchUtil.numpy_to_image(substraction_images_numpy[0])
            
            axes[0].set_title(f'{name} Low res: {predicted_image.shape}')
            axes[1].set_title(f'{name} High res: {interpolated_image.shape}')
            axes[2].set_title(f'{name} Predicted res: {subtraction_image.shape}')

            axes[0].imshow(predicted_image)
            axes[1].imshow(interpolated_image)
            axes[2].imshow(subtraction_image)

            plt.show()
            
            return

        for i in range(len(predicted_images_numpy)):
            predicted_image = torchUtil.numpy_to_image(predicted_images_numpy[i])
            interpolated_image = torchUtil.numpy_to_image(interpolated_images[i])
            subtraction_image = torchUtil.numpy_to_image(substraction_images_numpy[i])
            
            axes[i, 0].set_title(f'{name} Low res: {predicted_image.shape}')
            axes[i, 1].set_title(f'{name} High res: {interpolated_image.shape}')
            axes[i, 2].set_title(f'{name} Predicted res: {subtraction_image.shape}')

            axes[i, 0].imshow(predicted_image)
            axes[i, 1].imshow(interpolated_image)
            axes[i, 2].imshow(subtraction_image)

        plt.show()