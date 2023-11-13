import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class RDN(nn.Module):
    """Implementation of the Residual Dense Network for image super-scaling.

    The network is the one described in https://arxiv.org/abs/1802.08797 (Zhang et al. 2018).

    Args:
        c_dim: integer, number of channels of the input image.
        kernel_size: integer, common kernel size for convolutions.
        upscaling: string, 'ups' or 'shuffle', determines which implementation
            of the upscaling layer to use.
        weights: string, if not empty, load pre-trained weights.
            Overrides other parameters.

    Attributes:
        C: integer, number of conv layer inside each residual dense blocks (RDB).
        D: integer, number of RDBs.
        G: integer, number of convolution output filters inside the RDBs.
        G0: integer, number of output filters of each RDB.
        scaling_factor: integer, the scaling factor in scale.
    """
    
    def __init__(self, C , D, G ,G0, scaling_factor, kernel_size=3, c_dims=3, upscaling='ups', weights=None):
        super(RDN, self).__init__()
        self.D = D
        self.G = G
        self.G0 = G0
        self.C = C
        self.scale = scaling_factor
        self.kernel_size = kernel_size
        self.c_dims = c_dims
        self.upscaling = upscaling
        
        if weights:
            pass
            #TO DO - load weights

        self._prepare()
        
    
    def _prepare(self):
        """Prepare the network for training by initializing weights."""
        self.F_m1 = nn.Conv2d(in_channels=self.c_dims, out_channels=self.G0, kernel_size=self.kernel_size, padding='same') #F_m1 = F minus 1
        self.F_0 = nn.Conv2d(in_channels=self.G0, out_channels=self.G0, kernel_size=self.kernel_size, padding='same')
        self.F = []
        for d in range(self.D):
            self.F.append(self._make_RDBs())
        
        self.GFF1 = nn.Conv2d(in_channels=self.G0 + self.D * self.G, out_channels=self.G0, kernel_size=1, padding='same')
        self.GFF2 = nn.Conv2d(in_channels=self.G0, out_channels=self.G0, kernel_size=3, padding='same')

        self.FU = self._make_upsampling()

        self.F_last = nn.Conv2d(in_channels=self.c_dims * self.scale ** 2, out_channels=self.c_dims, kernel_size=self.kernel_size, padding='same')
        
        

    def _make_RDBs(self):
        """Builds the residual dense blocks."""
        rl = [] #rl for residual layers
        for c in range(1 , self.C + 1):
            rl.append(nn.Conv2d(in_channels=self.G0 + (c-1)*self.G, out_channels=self.G, kernel_size=self.kernel_size, padding='same'))

        rl.append(nn.Conv2d(in_channels=self.G0 + self.C*self.G, out_channels=self.G, kernel_size=1, padding='same')) #Local feature fusion
        return rl
    

    def _make_upsampling(self):
        """return an upsampling function that multiply per scale the height and the weight of the image."""
        # return F.interpolate #maybe add conv to learn the upsampling
        if self.upscaling == 'shuffle':
            UPN1 = nn.Conv2d(in_channels=self.G0, out_channels=64, kernel_size=5, padding='same')
            UPN1_Relu = nn.ReLU()
            UPN2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
            UPN2_Relu = nn.ReLU()
            UPN3 = nn.Conv2d(in_channels=32, out_channels=self.c_dims * self.scale ** 2, kernel_size=3, padding='same')
            #TODO: add shuffle
            pass
            return nn.Sequential(UPN1, UPN1_Relu, UPN2, UPN2_Relu, UPN3)
        elif self.upscaling == 'ups':
            return nn.Sequential(
                nn.Conv2d(in_channels=self.G0, out_channels=64, kernel_size=5, padding='same'),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=self.c_dims * self.scale ** 2, kernel_size=3, padding='same')
            )
        else:
            raise ValueError('Invalid choice of upscaling layer.')


    def forward(self, I_LR):
        """Forward pass of the network.

        Args:
            I_LR: input image, tensor of shape (N, C, H, W).

        Returns:
            H_LR: output image, tensor of shape (N, C, scale*H, scale*W).
        """
        
        f_m1 = F.relu(self.F_m1(I_LR))
        f_0 = F.relu(self.F_0(f_m1))
        f = f_0
        f_tmp = f_0
        for d in range(0, self.D):
            f_d_c = f_tmp
            for c in range(0, self.C):
                f_d_c_tmp = F.relu(self.F[d][c](f_d_c))
                f_d_c = torch.cat((f_d_c, f_d_c_tmp), dim=0)
            f_d_LF = F.relu(self.F[d][self.C](f_d_c))
            f_tmp = f_d_LF + f_tmp
            f = torch.cat((f, f_tmp), dim=0)
        
        f_GF = self.GFF1(f)
        f_GF = self.GFF2(f_GF)
        
        f_DF = f_m1 + f_GF
        
        f_U = self.FU(f_DF)
        if self.upscaling == 'ups':
            f_U = f_U.unsqueeze(1)
            f_U = F.interpolate(f_U, scale_factor=self.scale, mode='bilinear', align_corners=False)
            f_U = f_U.squeeze(1)
        I_HR = torch.sigmoid(self.F_last(f_U))
        return I_HR