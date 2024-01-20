from src.models.BaseNN import BaseNN
from torch import nn, concat

class UpscaleResidual2NN(BaseNN):
	def __init__(self, default_upscale_factor=None,
			  num_channel: int=3, channel_interpolation: list = None,
			  old_version: bool=False) -> None:
		
		super().__init__(default_upscale_factor, 
            num_channel=num_channel, channel_interpolation=channel_interpolation,
            old_version=old_version)
		
		self.encod1 = nn.Sequential(
            self.DoubleConv2d(num_channel, 16),
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
        
		self.encod4 = nn.Sequential(
            self.DoubleConv2d(64, 128),
            nn.BatchNorm2d(128)
        )
        
		self.decod1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
		self.decod2 = nn.Sequential(
            self.DoubleConv2d(128, 64),
            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
		self.decod3 = nn.Sequential(
            self.DoubleConv2d(64, 32),
            nn.ConvTranspose2d(32, 16, 2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        
		self.decod4 = nn.Sequential(
            self.DoubleConv2d(32, 16),
            nn.Conv2d(16, 3, 1),
        )

	
	def forward(self, X):
		X_U = self.upscale_image(X)

        # take only the first 3 channels
		if len(X_U.shape) == 4:
			X_U_resid = X_U[:, :3, :, :]
		else:
			X_U_resid = X_U[:3, :, :]

		X_1 = self.encod1(X_U)

		X_2 = self.encod2(nn.MaxPool2d(2)(X_1))

		X_3 = self.encod3(nn.MaxPool2d(2)(X_2))

		X_4 = self.encod4(nn.MaxPool2d(2)(X_3))

		result = self.decod1(X_4)
		result = self.decod2(concat((X_3, result), dim = 1))
		result = self.decod3(concat((X_2, result), dim = 1))
		result = self.decod4(concat((X_1, result), dim = 1))
        
		return (result + X_U_resid).clamp(0,1)
