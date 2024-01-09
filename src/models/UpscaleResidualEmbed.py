from src.models.UpscaleNN import UpscaleNN

import torch
from torch import nn, concat

class UpscaleResidualEmbedNN(UpscaleNN):
	def __init__(self, device, default_upscale_factor=None, architecture=None, old_version=False) -> None:
		super().__init__(device, default_upscale_factor, old_version=old_version)
		
		if architecture is None:
			architecture = [16, 32, 64]


		self.encods.append(nn.Sequential(
			self.DoubleConv2d(3 + 1, architecture[0]),
			nn.BatchNorm2d(architecture[0])
		))

		self.encod1 = nn.Sequential(
			self.DoubleConv2d(4, 16),
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
		# Add a canl to X_U' full of self.upscale_factor_tensor (tensor) or self.upscale_factor(tensor) (so 3 -> 4)

		new_channel = torch.full((X_U.shape[0], 1, X_U.shape[2], X_U.shape[3]), self.upscale_factor, device=self.device)
		X_u_c = torch.cat((X_U, new_channel), dim=1)

		#print(X_U.shape, X_u_c.shape, new_channel.shape)
		X_1 = self.encod1(X_u_c)

		X_2 = self.encod2(nn.MaxPool2d(2)(X_1))

		X_4 = self.encod3(nn.MaxPool2d(2)(X_2))

		result = self.decod1(X_4)
		result = self.decod2(concat((X_2, result), dim = 1))
		result = self.decod3(concat((X_1, result), dim = 1))

		# remove the last channel (4 -> 3)
		#result = result[:, :-1, :, :]
		
		return (result + X_U).clamp(0,1)
