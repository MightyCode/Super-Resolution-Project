# Super-Resolution-Project
S9 enseirb IA project for super resolution

| Type entrainement / modèles | Train PSNR-SSIM x2 | Train  PSNR-SSIM x4 | Train PSNR-SSIM x8 | Test PSNR-SSIM x2 | Test  PSNR-SSIM x4 | Test PSNR-SSIM x8 |
| --------------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Nearsest | 33.80 - 0.917 | 28.58 - 0.788 | 24.66 - 0.640 | 33.61 - 0.913 | 28.44 - 0.78 | 24.59 - 0.636 |
| Billinear | 35.02 - 0.918 | 30.33 - 0.820 | 26.33 - 0.701 |  34.76 - 0.913 | 30.16 - 0.814 | 26.25 - 0.697 |
| Bicubic | 36.53 - 0.935 | 30.70 - 0.830 | 26.25 - 0.698  | 36.20 - 0.931 | 30.50 - 0.823 | 26.17 - 0.694  |
| Modèle de référence | X - X | X - X | X - X | X - X | X - X | X - X |
| Upscale residual lpips - image - x2 | 38.04 - 0.943 | 30.62 - 0.828 | 26.23 - 0.697  | 37.65 - 0.938 | 30.34 - 0.820 | 26.14 - 0.693 |
| Upscale residual lpips - image - x2 x4 x8  | 38.79 - 0.952 | 32.07 - 0.86 | 27.37 - 0.736 | 38.73 - 0.961 | 32.47 - 0.892 | 27.96 - 0.798 |
| Modèle - image/Patch - x2 - Modalitées Carla  | X - X | X - X | X - X | X - X | X - X | X - X |
| Modèle - (image/Patch) - x2 x4 x8 - Modalitées Carla | X - X | X - X | X - X | X - X | X - X | X - X |