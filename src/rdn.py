import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class RDN(nn.Module):
    """Implementation of the Residual Dense Network for image super-scaling.

    The network is the one described in https://arxiv.org/abs/1802.08797 (Zhang et al. 2018).

    Args:
        arch_params: dictionary, contains the network parameters C, D, G, G0, scale.
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
        name: name used to identify what upscaling network is used during training.
        model._name: identifies this network as the generator network
            in the compound model built by the trainer class.
    """
    
    def __init__(self, C , D, G ,G0, scaling_factor, kernel_size=3, c_dims=3, upscaling='ups', weights=None):
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
        
    

    def prepare(self):
        """Prepare the network for training by initializing weights."""
        self.F_m1 = nn.Conv2d(self.c_dims, self.G0, kernel_size=self.kernel_size, padding='same')
        self.F_0 = nn.Conv2d(self.c_dims, self.G0, kernel_size=self.kernel_size, padding='same')
        self.F_D = self._make_residual_blocks()
        self.GFF1 = nn.Conv2d(self.c_dims, self.G0, kernel_size=1, padding='same')
        self.GFF2 = nn.Conv2d(self.c_dims, self.G0, kernel_size=self.kernel_size, padding='same')
        self.FU = self._UPN()
        self.SR = nn.Conv2d(self.c_dims, self.c_dims, kernel_size=self.kernel_size, padding='same')
        
        
    
    def _make_residual_blocks(self):
        blocks = []
        for _ in range(self.D):
            blocks.append(self._make_residual_block())
        return nn.Sequential(*blocks)
    
    
    def _make_residual_block(self):
        conv_relu = []
        for _ in range(self.C):
            conv_relu.append(nn.Conv2d(self.c_dims, self.G, kernel_size=self.kernel_size, padding='same'))
            conv_relu.append(nn.ReLU(inplace=True))
        return nn.Sequential(*conv_relu)
    
    def _UPN(self):
        """ Upscaling network """
        if self.upscaling == 'ups':
            return nn.Sequential(
                nn.Conv2d(self.c_dims, 64, kernel_size=5, strides=1, padding='same'),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.c_dims, 32, kernel_size=5, strides=1, padding='same'),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.c_dims, self.c_dims * self.scale ** 2, kernel_size=self.kernel_size, padding='same'),
                nn.PixelShuffle(self.scale),
            )
        elif self.upscaling == 'shuffle':
            return nn.Sequential(
                nn.Conv2d(self.c_dims, 64, kernel_size=5, strides=1, padding='same'),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.c_dims, 32, kernel_size=5, strides=1, padding='same'),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(self.scale),
            )
        else:
            raise ValueError("Invalid upscaling method selected")
    
    def forward(self, x):
        """Forward pass of the network.

        Args:
            x: input image, tensor of shape (N, C, H, W).

        Returns:
            x: output image, tensor of shape (N, C, H, W).
        """
        x = self.F_m1(x)
        x1 = self.F_0(x)
        x = self.F_D(x1)
        x = self.GFF1(x) + x1
        x = self.FU(x)
        x = self.SR(x)
        return x