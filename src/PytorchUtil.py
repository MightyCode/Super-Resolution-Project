import torch
import cv2
import numpy as np

class PytorchUtil:
    @staticmethod
    def tensor_to_numpy(tensor):
        return tensor.detach().cpu().numpy().transpose(1, 2, 0)

    def tensor_to_image(tensor, color_mode=cv2.COLOR_BGR2RGB):
        return cv2.cvtColor(PytorchUtil.tensor_to_numpy(tensor), color_mode)

    def numpy_to_tensor(numpy):
        return torch.from_numpy(numpy.astype(np.float32).transpose(2, 0, 1))
        