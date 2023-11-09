import os, sys
from skimage.metrics import mean_squared_error

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent_dir)

if __name__ == "__main__":
    import cv2
    from metrics import Metric

    path1 = "resources/1.png" if len(sys.argv) < 2 else sys.argv[1]
    image = cv2.imread(path1)

    inverted_image = cv2.bitwise_not(image)
    

    print(f"MSE : {Metric.MSE(image, inverted_image)} | SKImage MSE : {mean_squared_error(image, inverted_image)}")
    print(f"PSNR : {Metric.PSNR(image, inverted_image)}")
    print(f"SSMH : {Metric.SSMH(image, inverted_image)}")
    print(f"SSIM : {Metric.SSIM(image, inverted_image)} | SKimage SSIM : {Metric.SSIM_ref(image, inverted_image)}")
    print(f"SSIM : {Metric.SSIM_2(image, inverted_image)} | SKimage SSIM : {Metric.SSIM_ref(image, inverted_image)}")

    
