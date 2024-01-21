# Super-Resolution-Project
S9 enseirb IA project for super resolution

## Dataset 1 : images (RGB)

| Type entrainement / modèles | Train PSNR-SSIM x2 | Train  PSNR-SSIM x4 | Train PSNR-SSIM x8 | Test PSNR-SSIM x2 | Test  PSNR-SSIM x4 | Test PSNR-SSIM x8 |
| --------------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Nearsest | 33.80 - 0.917 | 28.58 - 0.788 | 24.66 - 0.640 | 33.61 - 0.913 | 28.44 - 0.78 | 24.59 - 0.636 |
| Billinear | 35.02 - 0.918 | 30.33 - 0.820 | 26.33 - 0.701 |  34.76 - 0.913 | 30.16 - 0.814 | 26.25 - 0.697 |
| Bicubic | 36.53 - 0.935 | 30.70 - 0.830 | 26.25 - 0.698  | 36.20 - 0.931 | 30.50 - 0.823 | 26.17 - 0.694  |
| Modèle de référence | X - X | X - X | X - X | X - X | X - X | X - X |
| Upscale residual lpips - image - x2 | 38.04 - 0.943 | 30.62 - 0.828 | 26.23 - 0.697  | 37.65 - 0.938 | 30.34 - 0.820 | 26.14 - 0.693 |
| Upscale residual lpips - image - x2 x4 x8  | 38.81 - 0.953 | 32.13 - 0.86 | 27.48 - 0.738 | 38.80 - 0.962 | 32.54 - 0.893 | 28.10 - 0.800 |

## Dataset 2 : images (RGBDS)

D is for Depth, S is for segmentation.

| Type entrainement / modèles | Train PSNR-SSIM x2 | Train  PSNR-SSIM x4 | Train PSNR-SSIM x8 | Test PSNR-SSIM x2 | Test  PSNR-SSIM x4 | Test PSNR-SSIM x8 |
| --------------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Nearsest | 33.80 - 0.917 | 28.58 - 0.788 | 24.66 - 0.640 | 33.61 - 0.913 | 28.44 - 0.78 | 24.59 - 0.636 |
| Billinear | 35.02 - 0.918 | 30.33 - 0.820 | 26.33 - 0.701 |  34.76 - 0.913 | 30.16 - 0.814 | 26.25 - 0.697 |
| Bicubic | 36.53 - 0.935 | 30.70 - 0.830 | 26.25 - 0.698  | 36.20 - 0.931 | 30.50 - 0.823 | 26.17 - 0.694  |
| Modèle de référence | X - X | X - X | X - X | X - X | X - X | X - X |
| Upscale residual lpips - image - x2 | 38.04 - 0.943 | 30.62 - 0.828 | 26.23 - 0.697  | 37.65 - 0.938 | 30.34 - 0.820 | 26.14 - 0.693 |
| Upscale residual lpips - image - x2 x4 x8  | 38.81 - 0.953 | 32.13 - 0.86 | 27.48 - 0.738 | 38.80 - 0.962 | 32.54 - 0.893 | 28.10 - 0.800 |
| Modèle - image/Patch - x2 - Modalitées Carla  | X - X | X - X | X - X | X - X | X - X | X - X |
| Modèle - (image/Patch) - x2 x4 x8 - Modalitées Carla | X - X | X - X | X - X | X - X | X - X | X - X |


## Nommage des données

hr => donnée ou image haute résolution
lr => donnée ou image base résolution

_data => donnée d'entrées que nous donnerai une opération un dataset[x]
_img => ensuite si on est sur de retirer les canaux superflux des "data" ou que le modèle a forward la donnée, qui devient alors forcément une image

_patch => si la donnée concerne un patch d'une donnée globale

_tensor => si la donnée est un tenseur
_np => si la donnée est un numpy

+(s) si la donnée est une liste 

**Examples*

hr_data_patch_tensors => une liste de patch de données représentés sous forme de tenseur

lr_img_np => une image représenté sous un tableau numpy

