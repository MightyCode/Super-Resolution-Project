from torch import nn
from torchvision.transforms.v2 import Resize
from torchvision import transforms

class UpscaleNN(nn.Module):
	def __init__(self, default_upscale_factor=None, old_version=False) -> None:
		super().__init__()
		self.upscale_factor = default_upscale_factor

		self.old_version = old_version

		self.encoder = nn.Sequential(
			self.DoubleConv2d(3, 9),
			nn.MaxPool2d(2),
			self.DoubleConv2d(9, 27),
			nn.MaxPool2d(2),
			self.DoubleConv2d(27, 54),
		)
		
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(54, 27, 2, 2),
			self.DoubleConv2d(27,27),
			nn.ConvTranspose2d(27, 9, 2, 2),
			self.DoubleConv2d(9, 9),
		)

		self.final = nn.Conv2d(9, 3, 1)

	def __str__(self) -> str:
		return __class__.__name__ + f"(upscale_factor={self.upscale_factor})"

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
		else:
			return Resize((new_height, new_width), interpolation=transforms.InterpolationMode.BICUBIC, 
				antialias=True)(image)

	def forward(self, X):
		X_2 = self.upscale_image(X)

		value = self.final(self.decoder(self.encoder(X_2))) + X_2

		# Clamp value beteween 0 and 1
		return value.clamp(0, 1)

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
	
