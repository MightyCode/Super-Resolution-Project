from torch import nn, concat
from torchvision.transforms.v2 import Resize
from torchvision import transforms

class UpscaleNN(nn.Module):
	def __init__(self, super_res_factor=2, old_version=False) -> None:
		super().__init__()
		self.super_res_factor = super_res_factor

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

	
	def upscale_image(self, image):
		if len(image.shape) == 3:
			_, h, w = image.shape
			image = image.unsqueeze(0)
		else:
			_, _, h, w = image.shape

		new_width = int(w * self.super_res_factor)
		new_height = int(h * self.super_res_factor)

		if self.old_version:
			return Resize((new_height, new_width), interpolation=transforms.InterpolationMode.BILINEAR)(image)
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
	
class UpscaleResidualNN(UpscaleNN):
	def __init__(self, super_res_factor=2, old_version=False) -> None:
		super().__init__(super_res_factor, old_version=old_version)
		self.super_res_factor = super_res_factor
		
		self.encod1 = nn.Sequential(
			self.DoubleConv2d(3, 16),
			nn.BatchNorm2d(16)
		)
		self.encod2 = nn.Sequential(
			self.DoubleConv2d(16, 32),
			nn.BatchNorm2d(32)
		)
		self.encod3 = nn.Sequential(
			self.DoubleConv2d(32, 64),
			nn.BatchNorm2d(64)
		)
		self.decod1 = nn.Sequential(
			nn.ConvTranspose2d(64, 32, 2, 2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
		)
		self.decod2 = nn.Sequential(
			self.DoubleConv2d(64, 32),
			nn.ConvTranspose2d(32, 16, 2, 2),
			nn.BatchNorm2d(16),
			nn.ReLU(),
		)
		self.decod3 = nn.Sequential(
			self.DoubleConv2d(32, 16),
			nn.Conv2d(16, 3, 1),
		)

	
	def forward(self, X):
		X_U = self.upscale_image(X)

		X_1 = self.encod1(X_U)

		X_2 = self.encod2(nn.MaxPool2d(2)(X_1))

		X_4 = self.encod3(nn.MaxPool2d(2)(X_2))

		result = self.decod1(X_4)
		result = self.decod2(concat((X_2,result), dim = 1))
		result = self.decod3(concat((X_1,result), dim = 1))
		
		return (result + X_U).clamp(0,1)
