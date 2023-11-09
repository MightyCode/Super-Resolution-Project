class Damager:
    @staticmethod
    def add_gaussian_noise(image, mean=0, std=25):
        noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        
        return noisy_image

    @staticmethod
    def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
        noisy_image = image.copy()
        total_pixels = image.size
        salt_pixels = int(total_pixels * salt_prob)
        pepper_pixels = int(total_pixels * pepper_prob)

        salt_coords = [np.random.randint(0, i - 1, salt_pixels) for i in image.shape]
        pepper_coords = [np.random.randint(0, i - 1, pepper_pixels) for i in image.shape]

        noisy_image[salt_coords[0], salt_coords[1]] = 255
        noisy_image[pepper_coords[0], pepper_coords[1]] = 0

        return noisy_image

    @staticmethod
    def apply_blur(image, kernel_size=(5, 5)):
        return cv2.GaussianBlur(image, kernel_size, 0)

    @staticmethod
    def introduce_compression_artifacts(image, quality=10):
        _, encoded_image = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(encoded_image, 1)



if __name__ == "__main__":
    import cv2
    import numpy as np
    import sys, os
    import matplotlib.pyplot as plt

    # Load image
    path = "resources/im1.jpg" if len(sys.argv) < 2 else sys.argv[1]
    
    output_directory = "results"  # Default output directory
    if len(sys.argv) > 2:
        output_directory = os.path.join(output_directory, sys.argv[2])
    
    os.makedirs(output_directory, exist_ok=True)

    base_filename, file_extension = os.path.splitext(os.path.basename(path))
    result_filename = f"{base_filename}-alt{file_extension}"
    result_path = os.path.join(output_directory, result_filename)

    if not os.path.exists(path):
        print(f"Path not existing : {path} \n Aborting ...")
        exit()

    image = cv2.imread(path)
    cv2.imshow('Noisy Image', image)

    damaged_image = Damager.add_salt_and_pepper_noise(Damager.apply_blur(image))

    cv2.imshow('Noisy Image', damaged_image)
    print(f"Saving images to {result_path}")
    cv2.imwrite(result_path, damaged_image)

    rows = 1
    columns = 2

    fig = plt.figure(figsize=(10, 7))  

    fig.add_subplot(rows, columns, 1) 
    plt.imshow(np.fliplr(image.reshape(-1,3)).reshape(image.shape))
    plt.axis('off') 
    plt.title("Original image")  

    fig.add_subplot(rows, columns, 2) 
    plt.imshow(np.fliplr(damaged_image.reshape(-1,3)).reshape(damaged_image.shape)) 
    plt.axis('off') 
    plt.title("Quality reduced")  

    from metrics import Metric

    mse = Metric.MSE(image, damaged_image)
    pscn = Metric.PSNR(image, damaged_image)
    ssmh = Metric.SSMH(image, damaged_image)
    ssim = Metric.SSIM(image, damaged_image)

    plt.text(-360, 600, f"MSE : {round(mse, 3)}", ha="center")
    plt.text(-150, 600, f"PSCN : {round(pscn, 3)}", ha="center")
    plt.text(80, 600, f"SSMH : {round(ssmh, 3)}", ha="center")
    plt.text(260, 600, f"SSIM : {round(ssim, 3)}", ha="center")

    plt.show()