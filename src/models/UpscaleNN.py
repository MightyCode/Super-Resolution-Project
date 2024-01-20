from src.models.BaseNN import BaseNN

from torch import nn


class UpscaleNN(BaseNN):
	def __init__(self, default_upscale_factor=None,
			  num_channel: int=3, channel_interpolation: list = None,
			  old_version: bool=False) -> None:
		
		super().__init__(default_upscale_factor, 
            num_channel=num_channel, channel_interpolation=channel_interpolation,
            old_version=old_version)

		self.encoder = nn.Sequential(
			self.DoubleConv2d(num_channel, 9),
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

	def forward(self, X):
		X_2 = self.upscale_image(X)

		value = self.final(self.decoder(self.encoder(X_2))) + X_2

		# Clamp value beteween 0 and 1
		return value.clamp(0, 1)
	
