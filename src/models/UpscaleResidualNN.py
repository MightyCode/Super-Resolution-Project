from src.models.UpscaleNN import UpscaleNN

from torch import nn, concat

class UpscaleResidualNN(UpscaleNN):
	def __init__(self, default_upscale_factor=None, old_version=False) -> None:
		super().__init__(default_upscale_factor, old_version=old_version)
		
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
		result = self.decod2(concat((X_2, result), dim = 1))
		result = self.decod3(concat((X_1, result), dim = 1))
		
		return (result + X_U).clamp(0,1)
