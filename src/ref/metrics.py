import numpy as np
import cv2
import skimage

class Metric:
    @staticmethod
    def MSE(image1, image2):
        if image1.shape != image2.shape:
            raise ValueError("Images must have the same dimensions.")

        mse = np.mean((image1 - image2) ** 2, dtype=np.float64)

        return mse

    @staticmethod
    def PSNR(image1, image2):
        if image1.shape != image2.shape:
            raise ValueError("Images must have the same dimensions.")
        
        image1_f = image1.astype(float) * 255.0
        image2_f = image2.astype(float) * 255.0

        mse = Metric.MSE(image1_f, image2_f)
        if mse == 0:
            return float('inf')
        
        max_pixel = max(np.max(image1_f), np.max(image2_f))
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

        """
        max_pixel = 255.0
        psnr = 10 * np.log10(max_pixel**2 / mse))
        """

        return psnr

    @staticmethod
    def SSMH(image1, image2):
        """
        Calculates the Structural Similarity Measure for Hue (SSMH) between two images in the HSV color space.

        Args:
            image1 (numpy.ndarray): The first image.
            image2 (numpy.ndarray): The second image.

        Returns:
            float: The SSMH value between the two images.
        """
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

    
    # Hand made SSIM
    @staticmethod
    def SSIM_2(image1, image2, K1=0.01, K2=0.03, L=255):
        # Calculate the Structural Similarity Measure (SSIM)
        # You can use OpenCV to convert the images to the YCrCb color space
        image1_ycrcb = cv2.cvtColor(image1, cv2.COLOR_BGR2YCrCb)
        image2_ycrcb = cv2.cvtColor(image2, cv2.COLOR_BGR2YCrCb)

        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        C3 = C2 / 2

        # Extract the Y channel
        y1 = image1_ycrcb[:, :, 0]
        y2 = image2_ycrcb[:, :, 0]

        # Calculate the mean and standard deviation of the Y channel
        mean1 = np.mean(y1)
        mean2 = np.mean(y2)
        std1 = np.std(y1)
        std2 = np.std(y2)

        # Calculate the covariance and correlation coefficient between the two images
        cov = np.mean((y1 - mean1) * (y2 - mean2))
        corr = cov / (std1 * std2)

        # Calculate the luminance, contrast and structure terms
        luminance = (2 * mean1 * mean2 + C1) / (mean1 ** 2 + mean2 ** 2 + C1)
        contrast = (2 * std1 * std2 + C2) / (std1 ** 2 + std2 ** 2 + C2)
        structure = (cov + C3) / (std1 * std2 + C3)

        # Calculate the SSIM value
        ssim = luminance * contrast * structure
        return ssim
    
    @staticmethod
    def SSIM(image1, image2, K1=0.03, K2=0.03, L=255):
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        C3 = C2 / 2

        image1 = np.array(image1, dtype=float)
        image2 = np.array(image2, dtype=float)

        mean1 = np.mean(image1)
        mean2 = np.mean(image2)

        # use variance unbiased as does skimage
        var1 = np.var(image1, ddof=0)
        var2 = np.var(image2, ddof=0)

        cov12 = np.mean((image1 - mean1) * (image2 - mean2))

        SSIM_L = (2 * mean1 * mean2 + C1) / (mean1**2 + mean2**2 + C1)
        SSIM_C = (2 * cov12 + C2) / (var1 + var2 + C2)
        SSIM_S = (cov12 + C3) / (np.sqrt(var1) * np.sqrt(var2) + C3)

        SSIM = SSIM_L * SSIM_C * SSIM_S

        return SSIM

    def SSIM_ref(image1, image2):
        # Compute SSIM
        return skimage.metrics.structural_similarity(image1, image2,
                                     win_size=7, data_range=255, multichannel=True, channel_axis=2)

if __name__ == "__main__":
    import sys
    from skimage.metrics import mean_squared_error
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr

    # Load image
    path1 = "resources/set14/baboon.png" if len(sys.argv) < 2 else sys.argv[1]
    path2 = "results/im1.jpg" if len(sys.argv) < 3 else sys.argv[2]
        
    image1 = cv2.imread(path1)
    # convert
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread(path2)
    # convert
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    print(f"MSE : {Metric.MSE(image1, image2)} | SKImage MSE : {mean_squared_error(image1, image2)}")
    print(f"PSNR : {Metric.PSNR(image1, image2)} | SKImage PSNR : {psnr(image1, image2, data_range=255)}")
    print(f"SSMH : {Metric.SSMH(image1, image2)}")
    print(f"SSIM : {Metric.SSIM(image1, image2)} | SKimage SSIM : {Metric.SSIM_ref(image1, image2)}")
    print(f"SSIM_2: {Metric.SSIM_2(image1, image2)}")