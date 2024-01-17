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
            ax[0].set_title(name + " Low resolution" + " (x" + \
                                                         str(upscales[0]) + ")" + str(low_res_image.shape))
            ax[1].imshow(high_res_image)
            ax[1].set_title(name + " High resolution (" + str(high_res_image.shape) + ")")

            plt.show()
            
            return

        for i in range(num_images):
            low_res_image = torchUtil.tensor_to_image(low_res_images[i])
            high_res_image = torchUtil.tensor_to_image(high_res_images[i])
            
            ax[i, 0].imshow(low_res_image)
            ax[i, 0].set_title(name + " Low resolution (" + str(i) + ")" + \
                               "(x" + str(upscales[i]) + ")" + str(low_res_image.shape))
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
                
                low_res_images.append(
                    dataset.filter_channels_to_image(low_res_patches[index_low_res_patch]))
                high_res_images.append(high_res)
                # torch to int
                upscale_factors.append(dataset.get_upscale_factor(index_low_res_patch))
            else:
                index = np.random.randint(len(dataset))
                low_res_patches, high_res = dataset[index]

                index_low_res_patch = np.random.randint(len(low_res_patches))
                
                low_res_images.append(
                    dataset.filter_channels_to_image(low_res_patches[index_low_res_patch]))
                high_res_images.append(high_res)
                upscale_factors.append(dataset.get_upscale_factor(index_low_res_patch))
        
        PlotUtils.show_high_low_res_images(low_res_images, high_res_images, upscales=upscale_factors,
                                               width=width, name=dataset.name())

    # plot Plot the loss, psnr and ssim curves in the second row
    @staticmethod
    def plot_training_curve_and_metric(exp): 
        history = exp.history

        training_history = history["training"]
        validation_history = history["validation"]

        if len(training_history) == 0:
            return

        upscale_factors = list(training_history[0].keys())
        # list of keys

        losses_key = list(training_history[0][upscale_factors[0]]["loss"].keys())
        number_loss = len(losses_key) + 1 if len(losses_key) > 1 else 1

        # Plot all losses first
        for i in range(number_loss - 1):
            if (losses_key[i] == "loss") and i > 0:
                # Swap 
                losses_key[0], losses_key[i] = losses_key[i], losses_key[0]

        if len(upscale_factors) == 1 and number_loss == 1:
            fig, axes = plt.subplots(1, 1, figsize=(5, 5))

            axes.plot([x[upscale_factors[0]]["loss"][losses_key[0]] for x in training_history], label="Training")
            axes.plot([x[upscale_factors[0]]["loss"][losses_key[0]] for x in validation_history], label="Validation")
            axes.set_title(f'Loss {losses_key[0]} (x{upscale_factors[0]})')
            
            axes.set_xlabel('Epoch')
            axes.set_ylabel('Loss')

            # set y axis in scientific notation
            axes.set_yscale('log')

            axes.legend()

        elif len(upscale_factors) == 1:
            fig, axes = plt.subplots(number_loss, 1, figsize=(5, 5 * number_loss))

            for i in range(number_loss):
                if i == 0 and number_loss > 1:
                    for i_p in range(number_loss - 1):
                        axes[i].plot([x[upscale_factors[0]]["loss"][losses_key[i_p]] for x in training_history], 
                                     label=("Train " + losses_key[i_p]))
                        axes[i].plot([x[upscale_factors[0]]["loss"][losses_key[i_p]] for x in validation_history], 
                                     label="Valid " + losses_key[i_p])
                    
                    axes[i].set_title(f'Cumulated loss (x{upscale_factors[0]})')
                else:
                    index = i - 1 if number_loss > 1 else i
                    loss_title = "total" if losses_key[index] == "loss" else losses_key[index]
                    axes[i].plot([x[upscale_factors[0]]["loss"][losses_key[index]] for x in training_history], label="Training")
                    axes[i].plot([x[upscale_factors[0]]["loss"][losses_key[index]] for x in validation_history], label="Validation")
                    axes[i].set_title(f'Loss {loss_title} (x{upscale_factors[0]})')
                
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel('Loss')

                # set y axis in scientific notation
                axes[i].set_yscale('log')

                axes[i].legend()
        elif len(losses_key) == 1:
            fig, axes = plt.subplots(len(upscale_factors), 1, figsize=(5 * len(upscale_factors), 5))

            for i in range(len(upscale_factors)):
                axes[i].plot([x[upscale_factors[i]]["loss"][losses_key[0]] for x in training_history], label="Training")
                axes[i].plot([x[upscale_factors[i]]["loss"][losses_key[0]] for x in validation_history], label="Validation")
                axes[i].set_title(f'Loss {losses_key[0]} (x{upscale_factors[i]})')
                
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel('Loss')

                # set y axis in scientific notation
                axes[i].set_yscale('log')

                axes[i].legend()
        else:
            # Plot the loss curves
            fig, axes = plt.subplots( number_loss, len(upscale_factors), figsize=(5 * len(upscale_factors), 5 * number_loss))

            for j in range(number_loss):
                for i in range(len(upscale_factors)):
                    if j == 0 and number_loss > 1:
                        for i_p in range(number_loss - 1):
                            axes[j, i].plot([x[upscale_factors[j]]["loss"][losses_key[i_p]] for x in training_history], 
                                            label="Train " + losses_key[i_p])
                            axes[j, i].plot([x[upscale_factors[j]]["loss"][losses_key[i_p]] for x in validation_history], 
                                            label="Valid " + losses_key[i_p])
                        
                        axes[j, i].set_title(f'Cumulated loss (x{upscale_factors[j]})')
                    else:
                        index = j - 1 if number_loss > 1 else j

                        loss_title = "total" if losses_key[index] == "loss" else losses_key[index]
                        axes[j, i].plot([x[upscale_factors[i]]["loss"][losses_key[index]] for x in training_history], label="Training")
                        axes[j, i].plot([x[upscale_factors[i]]["loss"][losses_key[index]] for x in validation_history], label="Validation")
                        axes[j, i].set_title(f'Loss {loss_title} (x{upscale_factors[i]})')
                    
                    axes[j, i].set_xlabel('Epoch')
                    axes[j, i].set_ylabel('Loss')

                    # set y axis in scientific notation
                    axes[j, i].set_yscale('log')

                    axes[j, i].legend()

        #fig.show()

        # Plot the metrics 

        metrics = list(training_history[0][upscale_factors[0]]["metric"].keys())

        if len(upscale_factors) == 1:
            fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 7 * len(metrics)))

            for i in range(len(metrics)):
                axes[i].plot([x[upscale_factors[0]]["metric"][metrics[i]] for x in training_history], 
                             label="Training")
                
                axes[i].plot([x[upscale_factors[0]]["metric"][metrics[i]] for x in validation_history], 
                             label="Validation")
                
                axes[i].set_title(f'{metrics[i]} (x{upscale_factors[0]})')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metrics[i])

                axes[i].set_yscale('log')

                axes[i].legend()
        else:
            fig, axes = plt.subplots(len(metrics), len(upscale_factors), figsize=(7 * len(metrics), 5 * len(upscale_factors)))
            
            for j in range(len(metrics)):
                for i in range(len(upscale_factors)):
                    axes[j, i].plot([x[upscale_factors[i]]["metric"][metrics[j]] for x in training_history], 
                                    label="Training")
                    
                    axes[j, i].plot([x[upscale_factors[i]]["metric"][metrics[j]] for x in validation_history], 
                                    label="Validation")
                    
                    axes[j, i].set_title(f'{metrics[j]} (x{upscale_factors[i]})')
                    axes[j, i].set_xlabel('Epoch')
                    axes[j, i].set_ylabel(metrics[j])

                    axes[j, i].set_yscale('log')

                    axes[j, i].legend()
        
        fig.show()

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

                low_res = dataset.filter_channels_to_image(low_res)
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
                
                chosen_upscale = i % dataset.number_upscale()

                low_res = low_res_patches[chosen_upscale]

                upscale = dataset.get_upscale_factor(chosen_upscale)

                model.net.set_upscale_mode(upscale)

                predicted_res = model.net(low_res.to(device))[0]

                low_res = dataset.filter_channels_to_image(low_res)
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

                chosen_upscale = i % dataset.number_upscale()
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

        _, axes = plt.subplots(len(low_image_tensors), 3, figsize=(30, 7 * len(low_image_tensors)))

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