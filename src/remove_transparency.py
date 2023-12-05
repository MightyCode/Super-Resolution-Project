import os
import cv2
import numpy as np

def process_images(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    files = os.listdir(input_folder)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    for image_file in image_files:
        # Load the image with alpha channel
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # Set pixels with alpha value 0 to black
        alpha_zero_indices = image[:, :, 3] == 0
        image[alpha_zero_indices] = [0, 0, 0, 255]

        # Set all alpha values to 255
        image[:, :, 3] = 255

        # Save the modified image to the output folder
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, image)

        print(f"Processed: {image_file}")

if __name__ == "__main__":
    input_folder = "resources/pokemon/sugimori/val/high_res"  # Replace with the path to your input folder
    output_folder = "results"  # Replace with the path to your output folder

    process_images(input_folder, output_folder)