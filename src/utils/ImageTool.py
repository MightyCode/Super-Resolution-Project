from src.utils.PytorchUtil import PytorchUtil as torchUtil

from skimage import metrics
import numpy as np
import torch

class ImageTool:
    """
    Class to compute metrics on images
    """
    @staticmethod
    def compute_metrics_dataset(model, dataloader, upscale_index, device, verbose=False):
        model.net.eval()
        
        with torch.no_grad():
            batch_size = dataloader.batch_size
            len_dataset = dataloader.dataset.__len__()
            
            psnr = np.zeros(len_dataset)
            ssim = np.zeros(len_dataset)

            for i, (low_res_batches, high_res_batch) in enumerate(dataloader):
                    low_res_batch = low_res_batches[upscale_index]
                    low_res_batch = low_res_batch.to(device)

                    predicted_images_batch = model.net(low_res_batch)

                    index = i * batch_size
                    end = min(index + batch_size, len_dataset)
                    
                    for j in range(0, end - index):
                        predicted = torchUtil.tensor_to_numpy(predicted_images_batch[j])
                        high_res = torchUtil.tensor_to_numpy(high_res_batch[j])

                        psnr[j + index] = metrics.peak_signal_noise_ratio(high_res, predicted)
                        ssim[j + index] = metrics.structural_similarity(high_res, predicted, win_size=7,
                                                                         data_range=1, multichannel=True, channel_axis=2)

                    # if verbose and every 20 %
                    if verbose and (index) % int(len_dataset / 20) == 0:
                        print("{}%".format(index / (len_dataset / 100)))

            return psnr, ssim