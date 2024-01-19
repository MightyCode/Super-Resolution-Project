from torch import nn
from torchvision.transforms.v2 import Resize
from torchvision import transforms

class BaseNN(nn.Module):
	def __init__(self, default_upscale_factor=None, num_channel: int=3, old_version: bool=False) -> None:
		super().__init__()
		self.upscale_factor = default_upscale_factor
		self.old_version = old_version
		self.num_channel = num_channel
    
	def get_num_channel(self):
		return self.num_channel

	def set_upscale_mode(self, upscale_factor):
		self.upscale_factor = upscale_factor

	def get_upscale_mode(self):
		return self.upscale_factor
	
	def upscale_image(self, image):
		if len(image.shape) == 3:
			_, h, w = image.shape
			image = image.unsqueeze(0)
		else:
			_, _, h, w = image.shape

		new_width = w * self.upscale_factor
		new_height = h * self.upscale_factor

		if self.old_version:
			return Resize((new_height, new_width), interpolation=transforms.InterpolationMode.BILINEAR, 
				antialias=True)(image)
		
		return Resize((new_height, new_width), interpolation=transforms.InterpolationMode.BICUBIC, 
				antialias=True)(image)
		
	def DoubleConv2d(self, c_in, c_out, k_size=3):
		if self.old_version:
			return nn.Sequential(
				nn.Conv2d(c_in, c_out, k_size, padding=1),
				nn.ReLU(),
				nn.Conv2d(c_out, c_out, k_size, padding=1),
				nn.ReLU()
			)
		else:
			return nn.Sequential(
				nn.Conv2d(c_in, c_out, k_size, padding=1, padding_mode="replicate"),
				nn.ReLU(),
				nn.Conv2d(c_out, c_out, k_size, padding=1, padding_mode="replicate"),
				nn.ReLU()
			)