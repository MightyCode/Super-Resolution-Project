import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.transforms.v2 import Resize

class PytorchUtil:
    """
    Convert a numpy array to a tensor
    Use .cpu() if case of using GPU
    """
    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().transpose(1, 2, 0)

    """
    Convert a tensor to an image, convert the bgr format used everywhere to rgb
    """
    @staticmethod
    def tensor_to_image(tensor: torch.Tensor, color_mode: int=cv2.COLOR_BGR2RGB) -> np.ndarray:
        return PytorchUtil.numpy_to_image(PytorchUtil.tensor_to_numpy(tensor), color_mode)

    """
    Convert a numpy array to a tensor
    """
    @staticmethod
    def numpy_to_image(numpy: np.ndarray, color_mode: int=cv2.COLOR_BGR2RGB) -> np.ndarray:
        return cv2.cvtColor(numpy, color_mode)

    """
    Convert a numpy array to a tensor
    """
    @staticmethod
    def numpy_to_tensor(numpy: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(numpy.astype(np.float32).transpose(2, 0, 1))
    
    """
    Convert an image with not so much color differences to a gray image with extrapolated color
    """
    @staticmethod
    def norm_numpy_image(image: np.ndarray) -> np.ndarray:
        vmin = image.min()
        vmax = image.max()

        return (image - vmin) / (vmax - vmin)


    """
    Resize the tensor to a certain size, using the interpolation method
    """
    @staticmethod
    def interpolation_str_to_method(interpolation: str) -> transforms.InterpolationMode:
        if interpolation == "bicubic" or interpolation == "cubic":
            return transforms.InterpolationMode.BICUBIC
        elif interpolation == "nearest":
            return transforms.InterpolationMode.NEAREST
        elif interpolation == "lanzcos":
            return transforms.InterpolationMode.LANCZOS
        elif interpolation == "linear" or interpolation == "bilinear":
            return transforms.InterpolationMode.BILINEAR
        else:
            raise ValueError(f"Interpolation not supported: {interpolation}")

    """
    Resize the tensor to a certain size, using the interpolation method
    interpolation can be a string or a list of string or a transforms.InterpolationMode method
    """
    @staticmethod
    def resize_tensor(tensor: torch.Tensor, size, interpolation = transforms.InterpolationMode.BICUBIC) -> torch.Tensor:
        if type(interpolation) == list:
            if len(tensor.shape) == 3:
                assert len(interpolation) == tensor.shape[0]
                result = torch.zeros((tensor.shape[0], size[0], size[1]), dtype=tensor.dtype, device=tensor.device)

                for i in range(tensor.shape[0]):
                    interpolation_method = interpolation[i]
                    if type(interpolation[i]) == str:
                        interpolation_method = PytorchUtil.interpolation_str_to_method(interpolation[i])
                    
                    result[i, :, :] = Resize(size, interpolation=interpolation_method, 
                                             antialias=True)(tensor[i, :, :]).clamp(0, 1)

                return result
            else:
                assert len(interpolation) == tensor.shape[1]

                result = torch.zeros((tensor.shape[0], tensor.shape[1], size[0], size[1]), dtype=tensor.dtype, device=tensor.device)

                for i in range(tensor.shape[1]):
                    interpolation_method = interpolation[i]
                    if type(interpolation[i]) == str:
                        interpolation_method = PytorchUtil.interpolation_str_to_method(interpolation[i])

                    result[:, i, :, :] = Resize(size, interpolation=interpolation_method,
                                                 antialias=True)(tensor[:, i, :, :]).clamp(0, 1)

                return result

        interpolation_method = interpolation
        if type(interpolation) == str:
            interpolation_method = PytorchUtil.interpolation_str_to_method(interpolation)

        return Resize(size, interpolation=interpolation_method, antialias=True)(tensor).clamp(0, 1)

    """
    Resize the tensor to size, using the interpolation method then return a numpy array
    """
    @staticmethod
    def resize_tensor_to_numpy(tensor: torch.Tensor, size, interpolation=transforms.InterpolationMode.BICUBIC) -> np.ndarray:
        return PytorchUtil.tensor_to_numpy(PytorchUtil.resize_tensor(tensor, size, interpolation))
    
    """
    Open the data at path, can be a png or a numpy array
    """
    @staticmethod
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

    """
    Shrink the data to new_res_size, using the channel in channel_used
    SO function used to resize to a smaller size the data
    """
    @staticmethod
    def shrink_data(img, new_res_size, channel_used="bgr", channel_position={"b" : 0, "g" : 1, "r": 0},
            channel_downresolution_methods=None) -> None:
        # our "image" can have multiple channels, so we need to resize each channel
        assert len(channel_used) <= len(channel_position.keys())

        result = np.zeros((new_res_size[1], new_res_size[0], img.shape[2]), dtype=np.uint8)

        for i in range(img.shape[2]):
            dest_channel = channel_position[channel_used[i]]
            method = cv2.INTER_AREA

            if channel_downresolution_methods is not None:
                wanted_method = channel_downresolution_methods[channel_used[i]]
                if wanted_method == "bicubic":
                    method = cv2.INTER_CUBIC
                elif wanted_method == "nearest":
                    method = cv2.INTER_NEAREST
                elif wanted_method == "area":
                    method = cv2.INTER_AREA
                elif wanted_method == "linear":
                    method = cv2.INTER_LINEAR
                else:
                    raise ValueError(f"Method not supported: {wanted_method}")

            result[:, :, dest_channel] = cv2.resize(img[:, :, i], new_res_size, interpolation=method)

        return result
    
    """
    Filter the data to only keep the channel in channel_used 
    Shape of the data: (height, width, channel) -> (width, height, 3) if numpy array
    Shape of the data: (channel, height, width) -> (3, height, width) if torch tensors
    """
    @staticmethod
    def filter_data_to_img(data, channel_position={"b" : 0, "g" : 1, "r": 0}, channel_used="bgr"):
        assert len(channel_used) <= len(channel_position.keys())

        # key only the channel in channel_used

        if len(channel_used) == len(channel_position.keys()):
            return data

        if type(data) == np.ndarray:
            result = np.zeros((data.shape[0], data.shape[1], len(channel_used)), dtype=data.dtype)
            count = 0
            for i, channel in enumerate(channel_used):
                if channel in channel_position.keys():
                    result[:, :, count] = data[:, :, i]

                    count += 1

            return result
        # else torch tensor
        elif type(data) == torch.Tensor:
            result = torch.zeros((len(channel_used), data.shape[1], data.shape[2]), dtype=data.dtype)
            count = 0
            # true false array
            
            for i, channel in enumerate(channel_used):
                if channel in channel_position.keys():
                    result[count, :, :] = data[i, :, :]

                    count += 1

            return result
            
        else:
            raise ValueError(f"Data type not supported: {type(data)}")
