from src.models.BaseNN import BaseNN
from torch import nn, concat

class UpscaleResidual3NN(BaseNN):
    def __init__(self, default_upscale_factor=None,
			  num_channel: int=3, channel_interpolation: list = None,
			  old_version: bool=False) -> None:
        
        super().__init__(default_upscale_factor, 
            num_channel=num_channel, channel_interpolation=channel_interpolation,
            old_version=old_version)

        dnum = num_channel * 2
        ddnum = dnum * 2
        dddnum = ddnum * 2

        self.encod1 = nn.Sequential(
            self.DoubleConv2d(num_channel, dnum),
            nn.BatchNorm2d(dnum)
        )
        self.encod2 = nn.Sequential(
            self.DoubleConv2d(dnum, ddnum),
            nn.BatchNorm2d(ddnum)
        )
        self.encod3 = nn.Sequential(
            self.DoubleConv2d(ddnum, dddnum),
            nn.BatchNorm2d(dddnum)
        )
        self.decod1 = nn.Sequential(
            nn.ConvTranspose2d(dddnum, ddnum, 2, 2),
            nn.BatchNorm2d(ddnum),
            nn.ReLU(),
        )
        self.decod2 = nn.Sequential(
            self.DoubleConv2d(dddnum, ddnum),
            nn.ConvTranspose2d(ddnum, dnum, 2, 2),
            nn.BatchNorm2d(dnum),
            nn.ReLU(),
        )
        self.decod3 = nn.Sequential(
            self.DoubleConv2d(ddnum, dnum),
            nn.Conv2d(dnum, 3, 1),
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

        X_4 = self.encod3(nn.MaxPool2d(2)(X_2))

        result = self.decod1(X_4)
        result = self.decod2(concat((X_2, result), dim = 1))
        result = self.decod3(concat((X_1, result), dim = 1))
        
        return (result + X_U_resid).clamp(0,1)
