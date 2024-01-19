from .PytorchUtil import PytorchUtil as torchUtil

import matplotlib.pyplot as plt
import numpy as np
import torch


class PlotUtils:
    """
    Works with dataset of data or patches of data
    """
    @staticmethod
    def show_high_low_res_images(lr_img_tensors: list, hr_img_tensors: list, 
                                 upscale_factor_list: list=None, 
                                 plot_width:int = 15, plot_title: str="") -> None:
        
        assert len(lr_img_tensors) == len(hr_img_tensors)

        num_images = len(lr_img_tensors)

        _, ax = plt.subplots(num_images, 2, figsize=(plot_width, 5 * num_images))

        if num_images == 1:
            lr_img = torchUtil.tensor_to_image(lr_img_tensors[0])
            hr_img = torchUtil.tensor_to_image(hr_img_tensors[0])
            
            ax[0].imshow(lr_img)
            ax[0].set_title(plot_title + " Low resolution" + " (x" + \
                                                         str(upscale_factor_list[0]) + ")" + str(lr_img.shape))
            ax[1].imshow(hr_img)
            ax[1].set_title(plot_title + " High resolution (" + str(hr_img.shape) + ")")

            plt.show()
            
            return

        for i in range(num_images):
            lr_img = torchUtil.tensor_to_image(lr_img_tensors[i])
            hr_img = torchUtil.tensor_to_image(hr_img_tensors[i])
            
            ax[i, 0].imshow(lr_img)
            ax[i, 0].set_title(plot_title + " Low resolution (" + str(i) + ")" + \
                               "(x" + str(upscale_factor_list[i]) + ")" + str(lr_img.shape))
            ax[i, 1].imshow(hr_img)
            ax[i, 1].set_title(plot_title + " High resolution (" + str(i) + ")" + str(hr_img.shape))

        plt.show()

    """
    Define either num_images so its random indices or img_indices list of indices
    Works with dataset of data or patches of data
    """
    @staticmethod
    def show_dataset_example(dataset,
                             num_images: int=1, img_indices: list=None, plot_width: int=15):
        lr_img_tensors = []
        hr_img_tensors = []
        upscale_factors = []

        num_images = max(num_images, len(img_indices) if img_indices else 0)

        for i in range(num_images):
            if img_indices:
                lr_data_tensors, hr_img_tensor = dataset[img_indices[i]]

                index_lr_patch = np.random.randint(len(lr_data_tensors))
                lr_data_patch_tensor = lr_data_tensors[index_lr_patch]
                lr_img_tensor = dataset.filter_channels_to_image(lr_data_patch_tensor)
                lr_img_tensors.append(lr_img_tensor)

                hr_img_tensors.append(hr_img_tensor)
                # torch to int
                upscale_factors.append(dataset.get_upscale_factor(index_low_res_patch))
            else:
                index = np.random.randint(len(dataset))
                lr_data_tensors, hr_img_tensor = dataset[index]

                index_low_res_patch = np.random.randint(len(lr_data_tensors))
                lr_data_patch_tensor = lr_data_tensors[index_low_res_patch]
                lr_img_tensor = dataset.filter_channels_to_image(lr_data_patch_tensor)
                lr_img_tensors.append(lr_img_tensor)
                

                hr_img_tensors.append(hr_img_tensor)
                upscale_factors.append(dataset.get_upscale_factor(index_low_res_patch))
        
        PlotUtils.show_high_low_res_images(lr_img_tensors, hr_img_tensors, upscale_factor_list=upscale_factors,
                                               plot_width=plot_width, plot_title=dataset.name())

    """
    Plot the loss, psnr and ssim curves in a second plot
    """
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

        fig.show()

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

    """
    Define either num_images so its random indices or img_indices list of indices
    Works with dataset of data or patches of data
    """
    @staticmethod
    def plot_images_from_model(model, dataset, device, 
                               num_images: int=1, indices:list=None):
        num_images = max(num_images, len(indices) if indices else 0)

        _, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

        if num_images == 1:
            axes[0].clear()
            axes[1].clear()
            axes[2].clear()

            with torch.no_grad():
                if indices:
                    lr_data_tensors, hr_img_tensor = dataset[indices[0]]
                    print("Chosen index", indices[0])
                else:
                    index = np.random.randint(len(dataset))
                    lr_data_tensors, hr_img_tensor = dataset[index]
                    print("Chosen index", index)

                chosen_upscale = 0
                lr_data_tensor = lr_data_tensors[chosen_upscale].to(device)

                model.net.set_upscale_mode(chosen_upscale)
                pred_img_tensor = model.net(lr_data_tensor)[0]

                lr_img_tensor = dataset.filter_channels_to_image(lr_data_tensor)
                lr_img = torchUtil.tensor_to_image(lr_img_tensor)
                hr_img = torchUtil.tensor_to_image(hr_img_tensor)
                
                pred_img = torchUtil.tensor_to_image(pred_img_tensor)

            axes[0].set_title(f'Low res (x{dataset.get_upscale_factor(lr_data_tensor)}): {lr_img.shape}')
            axes[1].set_title(f'High res: {hr_img.shape}')
            axes[2].set_title(f'Predicted res: {pred_img.shape}')

            axes[0].imshow(lr_img)
            axes[1].imshow(hr_img)
            axes[2].imshow(pred_img)
        
            plt.show()

            return

        for i in range(num_images):
            with torch.no_grad():
                if indices:
                    lr_data_tensors, hr_img_tensor = dataset[indices[i]]
                    print("Chosen index", indices[i])
                else:
                    index = np.random.randint(len(dataset))
                    lr_data_tensors, hr_img_tensor = dataset[index]
                    print("Chosen index", index)
                
                chosen_upscale = i % dataset.number_upscale()

                lr_data_tensor = lr_data_tensors[chosen_upscale].to(device)

                upscale = dataset.get_upscale_factor(chosen_upscale)
                model.net.set_upscale_mode(upscale)
                pred_img_tensor = model.net(lr_data_tensor)[0]

                lr_img_tensor = dataset.filter_channels_to_image(lr_data_tensor)
                lr_img = torchUtil.tensor_to_image(lr_img_tensor)

                hr_img = torchUtil.tensor_to_image(hr_img_tensor)

                pred_img = torchUtil.tensor_to_image(pred_img_tensor)

                axes[i, 0].set_title(f'Low res (x{upscale}): {lr_img.shape}')
                axes[i, 1].set_title(f'High res: {hr_img.shape}')
                axes[i, 2].set_title(f'Predicted res: {pred_img.shape}')

                axes[i, 0].imshow(lr_img)
                axes[i, 1].imshow(hr_img)
                axes[i, 2].imshow(pred_img)
        
        plt.show()

    @staticmethod
    def plot_predicted_and_bicubic(model, dataset, device, 
                                   num_images: int=1, indices: list=None):
        num_images = max(num_images, len(indices) if indices else 0)

        _, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

        if num_images == 1:
            axes[0].clear()
            axes[1].clear()
            axes[2].clear()

            with torch.no_grad():
                if indices:
                    lr_data_tensors, _ = dataset[indices[0]]
                    print("Chosen index", indices[0])
                else:
                    index = np.random.randint(len(dataset))
                    lr_data_tensors, _ = dataset[index]
                    print("Chosen index", index)

                chosen_upscale: int = 0
                lr_data_tensor = lr_data_tensors[chosen_upscale].to(device)
                upscale = dataset.get_upscale_factor(chosen_upscale)
                model.net.set_upscale_mode(upscale)
           
                pred_img_torch = model.net(lr_data_tensor)[0]

                pred_img_np = torchUtil.tensor_to_numpy(pred_img_torch)
                
                lr_img_tensor = dataset.filter_channels_to_image(lr_data_tensor)
                bicubic_img_np = torchUtil.resize_tensor_to_numpy(lr_img_tensor, (pred_img_np.shape[0], pred_img_np.shape[1]))
                subtraction_img_np = torchUtil.norm_numpy_image(pred_img_np - bicubic_img_np)

            print(subtraction_img_np.mean(), subtraction_img_np.std())

            axes[0].set_title(f'Predicted res (x{upscale}): {pred_img_np.shape}')
            axes[1].set_title(f'Bicubic res: {bicubic_img_np.shape}')
            axes[2].set_title(f'Substraction res: {subtraction_img_np.shape}')

            axes[0].imshow(torchUtil.numpy_to_image(pred_img_np))
            axes[1].imshow(torchUtil.numpy_to_image(bicubic_img_np))
            axes[2].imshow(torchUtil.numpy_to_image(subtraction_img_np), vmin=subtraction_img_np.min(), vmax=subtraction_img_np.max())  
        
            plt.show()

            return


        for i in range(num_images):
            with torch.no_grad():
                if indices:
                    lr_data_tensors, _ = dataset[indices[i]]
                    print("Chosen index", indices[i])
                else:
                    index = np.random.randint(len(dataset))
                    lr_data_tensors, _ = dataset[index]
                    print("Chosen index", index)

                chosen_upscale = i % dataset.number_upscale()
                lr_data_tensor = lr_data_tensors[chosen_upscale].to(device)
                upscale = dataset.get_upscale_factor(chosen_upscale)
                model.net.set_upscale_mode(upscale)
            
                pred_img_torch = model.net(lr_data_tensor)[0]

                pred_img_np = torchUtil.tensor_to_numpy(pred_img_torch)

                lr_img_tensor = dataset.filter_channels_to_image(lr_data_tensor)
                bicubic_img_np = torchUtil.resize_tensor_to_numpy(lr_img_tensor, (pred_img_np.shape[0], pred_img_np.shape[1]))
                subtraction_img_np = torchUtil.norm_numpy_image(pred_img_np - bicubic_img_np)

            print(subtraction_img_np.mean(), subtraction_img_np.std())

            axes[i, 0].set_title(f'Predicted res (x{upscale}): {pred_img_np.shape}')
            axes[i, 1].set_title(f'Bicubic res: {bicubic_img_np.shape}')
            axes[i, 2].set_title(f'Substraction res: {subtraction_img_np.shape}')

            axes[i, 0].imshow(torchUtil.numpy_to_image(pred_img_np), vmin=0, vmax=1)
            axes[i, 1].imshow(torchUtil.numpy_to_image(bicubic_img_np), vmin=0, vmax=1)
            axes[i, 2].imshow(torchUtil.numpy_to_image(subtraction_img_np), vmin=subtraction_img_np.min(), vmax=subtraction_img_np.max())  
        
        plt.show()

    
    @staticmethod
    def show_low_high_predicted(lr_img_tensors, hr_image_tensors, pred_img_tensors, plot_title=""):
        assert len(lr_img_tensors) == len(hr_image_tensors) == len(pred_img_tensors)

        _, axes = plt.subplots(len(lr_img_tensors), 3, figsize=(30, 7 * len(lr_img_tensors)))

        if len(lr_img_tensors) == 1:
            with torch.no_grad():
                lr_img = torchUtil.tensor_to_image(lr_img_tensors[0])
                hr_img = torchUtil.tensor_to_image(hr_image_tensors[0])
                pred_img = torchUtil.tensor_to_image(pred_img_tensors[0])
            
            axes[0].set_title(f'{plot_title} Low res: {lr_img.shape}')
            axes[1].set_title(f'{plot_title} High res: {hr_img.shape}')
            axes[2].set_title(f'{plot_title} Predicted res: {pred_img.shape}')

            axes[0].imshow(lr_img)
            axes[1].imshow(hr_img)
            axes[2].imshow(pred_img)

            plt.show()
            
            return

        for i in range(len(lr_img_tensors)):
            with torch.no_grad():
                lr_img = torchUtil.tensor_to_image(lr_img_tensors[i])
                hr_img = torchUtil.tensor_to_image(hr_image_tensors[i])
                pred_img = torchUtil.tensor_to_image(pred_img_tensors[i])
            
            axes[i, 0].set_title(f'{plot_title} Low res: {lr_img.shape}')
            axes[i, 1].set_title(f'{plot_title} High res: {hr_img.shape}')
            axes[i, 2].set_title(f'{plot_title} Predicted res: {pred_img.shape}')

            axes[i, 0].imshow(lr_img)
            axes[i, 1].imshow(hr_img)
            axes[i, 2].imshow(pred_img)

        plt.show()

    @staticmethod
    def show_predicted_interpolated_subtraction( 
            pred_img_nps: list, 
            interpolated_img_nps: list, 
            substraction_img_nps: list, 
            plot_title: str=""):
        
        assert len(pred_img_nps) == len(interpolated_img_nps) == len(substraction_img_nps)

        _, axes = plt.subplots(len(pred_img_nps), 3, figsize=(30, 5 * len(pred_img_nps)))

        if len(pred_img_nps) == 1:
            pred_img = torchUtil.numpy_to_image(pred_img_nps[0])
            interpolated_img = torchUtil.numpy_to_image(interpolated_img_nps[0])
            subtraction_img = torchUtil.numpy_to_image(substraction_img_nps[0])
            
            axes[0].set_title(f'{plot_title} Prediction res: {pred_img.shape}')
            axes[1].set_title(f'{plot_title} Interpolated res: {interpolated_img.shape}')
            axes[2].set_title(f'{plot_title} Subtraction res: {subtraction_img.shape}')

            axes[0].imshow(pred_img)
            axes[1].imshow(interpolated_img)
            axes[2].imshow(subtraction_img)

            plt.show()
            
            return

        for i in range(len(pred_img_nps)):
            pred_img = torchUtil.numpy_to_image(pred_img_nps[i])
            interpolated_img = torchUtil.numpy_to_image(interpolated_img_nps[i])
            subtraction_img = torchUtil.numpy_to_image(substraction_img_nps[i])
            
            axes[i, 0].set_title(f'{plot_title} Prediction res: {pred_img.shape}')
            axes[i, 1].set_title(f'{plot_title} Interpolated res: {interpolated_img.shape}')
            axes[i, 2].set_title(f'{plot_title} Subtraction res: {subtraction_img.shape}')

            axes[i, 0].imshow(pred_img)
            axes[i, 1].imshow(interpolated_img)
            axes[i, 2].imshow(subtraction_img)

        plt.show()