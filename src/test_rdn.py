from rdn import RDN
import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

# WEIGHTS_URLS = {
#     'psnr-large': {
#         'arch_params': {'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2},
#         'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rdn-C6-D20-G64-G064-x2/PSNR-driven/rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5',
#         'name': 'rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5'
#     },
#     'psnr-small': {
#         'arch_params': {'C': 3, 'D': 10, 'G': 64, 'G0': 64, 'x': 2},
#         'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rdn-C3-D10-G64-G064-x2/PSNR-driven/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5',
#         'name': 'rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5',
#     },
#     'noise-cancel': {
#         'arch_params': {'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2},
#         'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5',
#         'name': 'rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5',
#     }
# }


if __name__ == '__main__':
    img = Image.open('resources/pokemon_jpg/pokemon_jpg/1.jpg')
    plt.imshow(img)
    plt.show()
    img = torchvision.transforms.ToTensor()(img)
    print(img.shape)
    rdn = RDN(C=6 , D=9, G=64 ,G0=64, scaling_factor=2, kernel_size=3, c_dims=img.shape[0], upscaling='ups', weights=None)
    print(rdn)

    new_img = rdn(img)
    print(new_img.shape)
    new_img = new_img.detach().numpy()
    new_img = np.transpose(new_img, (1, 2, 0))
    print(new_img.shape)
    plt.imshow(new_img)
    plt.show()

