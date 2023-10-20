import numpy as np
import cv2

class Metric:
    @staticmethod
    def MSE(image1, image2):
        if image1.shape != image2.shape:
            raise ValueError("Images must have the same dimensions.")
        
        mse = np.mean((image1 - image2) ** 2) 

        return mse

    @staticmethod
    def PSCN(image1, image2):
        if image1.shape != image2.shape:
            raise ValueError("Images must have the same dimensions.")
        
        mse = Metric.MSE(image1, image2)
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

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

        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Mean of the input images
        mu1 = np.mean(image1_gray)
        mu2 = np.mean(image2_gray)

        # Variance of the input images
        var1 = np.var(image1_gray)
        var2 = np.var(image2_gray)

        # Covariance between the input images
        covar = np.cov(image1_gray, image2_gray, rowvar=False)[0, 1]

        # SSIM formula components
        numerator = (2 * mu1 * mu2 + C1) * (2 * covar + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (var1 + var2 + C2)

        # Calculate SSIM
        ssim = numerator / denominator

        return ssim


if __name__ == "__main__":
    import sys, os

    # Load image
    path1 = "resources/set14/baboon.png" if len(sys.argv) < 2 else sys.argv[1]
    path2 = "results/im1.jpg" if len(sys.argv) < 3 else sys.argv[2]
        
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)

    print(f"MSE : {Metric.MSE(image1, image2)}")
    print(f"PSCN : {Metric.PSCN(image1, image2)}")
    print(f"SSMH : {Metric.SSMH(image1, image2)}")
    print(f"SSIM : {Metric.SSIM(image1, image2)}")