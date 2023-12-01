from torch import nn
from torchvision.transforms.v2 import Resize

class UpscaleNN(nn.Module):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
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

	
	def forward(self, X):
		if len(X.shape) == 3:
			_, h, w = X.shape
			X = X.unsqueeze(0)
		else:
			_, _, h, w = X.shape
			
		X_2 = Resize((h*2, w*2))(X)

		value = self.final(self.decoder(self.encoder(X_2))) + X_2

		# Clamp value beteween 0 and 1
		return value.clamp(0, 1)
		
	def make_encoder(self, depth = 3, in_c = 3, mult = 3):
		seq = []
		for i in range(1, depth):
			seq.append(self.DoubleConv2d(in_c*i, in_c*(i+1)))
			seq.append(nn.MaxPool2d(2))
		seq = [self.DoubleConv2d(in_c, in_c*mult)]
			

	def DoubleConv2d(self, c_in, c_out, k_size=3):
		return nn.Sequential(
			nn.Conv2d(c_in, c_out, k_size, padding=1),
			nn.ReLU(),
			nn.Conv2d(c_out, c_out, k_size, padding=1),
			nn.ReLU()
		)