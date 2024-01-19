import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.transforms.v2 import Resize

class PytorchUtil:
    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().transpose(1, 2, 0)

    def tensor_to_image(tensor: torch.Tensor, color_mode: int=cv2.COLOR_BGR2RGB) -> np.ndarray:
        return PytorchUtil.numpy_to_image(PytorchUtil.tensor_to_numpy(tensor), color_mode)

    def numpy_to_image(numpy: np.ndarray, color_mode: int=cv2.COLOR_BGR2RGB) -> np.ndarray:
        return cv2.cvtColor(numpy, color_mode)

    def numpy_to_tensor(numpy: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(numpy.astype(np.float32).transpose(2, 0, 1))
        
    def norm_numpy_image(image: np.ndarray) -> np.ndarray:
        vmin = image.min()
        vmax = image.max()

        return (image - vmin) / (vmax - vmin)

    def resize_tensor(tensor: torch.Tensor, size, interpolation=transforms.InterpolationMode.BICUBIC) -> torch.Tensor:
        return Resize(size, interpolation=interpolation, antialias=True)(tensor).clamp(0, 1)

    def resize_tensor_to_numpy(tensor: torch.Tensor, size, interpolation=transforms.InterpolationMode.BICUBIC) -> np.ndarray:
        return PytorchUtil.tensor_to_numpy(PytorchUtil.resize_tensor(tensor, size, interpolation))
    
    def open_data(path):
        if path.endswith(".png"):
            img = cv2.imread(path)
        elif path.endswith(".npy"):
            # contains n channels
            img = np.load(path) 

            # invert the first and third channel
            img[:, :, [0, 2]] = img[:, :, [2, 0]]
        else:
            raise ValueError(f"Image format not supported: {path}")

        return img

    def resize_data(img, new_res_size, channel_used, channel_position) -> None:
        # our "image" can have multiple channels, so we need to resize each channel
        result = np.zeros((new_res_size[1], new_res_size[0], img.shape[2]), dtype=np.uint8)

        for i in range(img.shape[2]):
            dest_channel = channel_position[channel_used[i]]
            result[:, :, dest_channel] = cv2.resize(img[:, :, i], new_res_size, interpolation=cv2.INTER_CUBIC)

        return result

    def filter_data(data, channel_filter, channel_position):
        if type(data) == np.ndarray:
            return data[:, :, :3]
        # else torch tensor
        elif type(data) == torch.Tensor:
            return data[:3, :, :]
        else:
            raise ValueError(f"Data type not supported: {type(data)}")
