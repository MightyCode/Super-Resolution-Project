import numpy as np
import cv2
import skimage

class Metric:
    @staticmethod
    def MSE(image1, image2):
        if image1.shape != image2.shape:
            raise ValueError("Images must have the same dimensions.")
        
        image1_f = image1.astype(float)
        image2_f = image2.astype(float)

        mse = np.mean((image1_f - image2_f) ** 2, dtype=np.float64)

        return mse

    @staticmethod
    def PSNR(image1, image2):
        if image1.shape != image2.shape:
            raise ValueError("Images must have the same dimensions.")
        
        mse = Metric.MSE(image1, image2)
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

        """
        max_pixel = 255.0
        psnr = 10 * np.log10(max_pixel**2 / mse))
        """

        return psnr

    @staticmethod
    def SSMH(image1, image2):
        # Calculate the Structural Similarity Measure for Hue (SSMH)
        # You can use OpenCV to convert the images to the HSV color space
        image1_hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        image2_hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

        # Extract the hue channel
        hue1 = image1_hsv[:, :, 0]
        hue2 = image2_hsv[:, :, 0]

        # Calculate the mean and standard deviation of the hue difference
        mean_diff = np.mean(hue1 - hue2)
        std_diff = np.std(hue1 - hue2)

        # SSMH formula: SSMH = (2 * mean_diff * std_diff) / (mean_diff ** 2 + std_diff ** 2)
        ssmh = (2 * mean_diff * std_diff) / (mean_diff ** 2 + std_diff ** 2)
        return ssmh
    
    @staticmethod
    def SSIM(image1, image2, K1=0.01, K2=0.03, L=255):
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        C3 = C2 / 2

        # Convert the input images to numpy arrays
        image1 = np.array(image1, dtype=float)
        image2 = np.array(image2, dtype=float)

        # Calculate the means of the images
        mean1 = np.mean(image1)
        mean2 = np.mean(image2)

        # Calculate the variances of the images
        var1 = np.var(image1)
        var2 = np.var(image2)

        # Calculate the covariance
        cov12 = np.mean((image1 - mean1) * (image2 - mean2))

        # Calculate the SSIM components
        SSIM_L = (2 * mean1 * mean2 + C1) / (mean1**2 + mean2**2 + C1)
        SSIM_C = (2 * cov12 + C2) / (var1 + var2 + C2)
        SSIM_S = (cov12 + C3) / (np.sqrt(var1) * np.sqrt(var2) + C3)

        # Calculate the overall SSIM value
        SSIM = SSIM_L * SSIM_C * SSIM_S

        return SSIM

    def SSIM_ref(image1, image2):
        image1_gray = np.mean(image1, axis=2)
        image2_gray = np.mean(image2, axis=2)

        print(image2_gray)

        # Compute SSIM
        return skimage.metrics.structural_similarity(image1_gray, image2_gray, data_range=255)

if __name__ == "__main__":
    import sys

    # Load image
    path1 = "resources/set14/baboon.png" if len(sys.argv) < 2 else sys.argv[1]
    path2 = "results/im1.jpg" if len(sys.argv) < 3 else sys.argv[2]
        
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)

    print(f"MSE : {Metric.MSE(image1, image2)}")
    print(f"PSNR : {Metric.PSNR(image1, image2)}")
    print(f"SSMH : {Metric.SSMH(image1, image2)}")
    print(f"SSIM : {Metric.SSIM(image1, image2)} | SKimage SSIM : {Metric.SSIM_ref(image1, image2)}")