import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Load the pre-trained super resolution model from TFHub
model_url = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
model = hub.load(model_url)

# Function to perform super resolution on an image
def super_resolve_image(input_path, output_path):
    # Load the input image
    input_image = Image.open(input_path)
    
    # Convert the image to numpy array and normalize pixel values
    input_array = np.array(input_image) / 255.0
    
    # Add batch dimension to the input image
    input_array = np.expand_dims(input_array, axis=0)
    
    # Perform super resolution
    super_resolved = model(input_array)
    
    # Denormalize pixel values
    super_resolved = super_resolved.numpy() * 255.0
    
    # Convert numpy array back to image
    super_resolved_image = Image.fromarray(super_resolved[0].astype(np.uint8))
    
    # Save the super resolved image
    super_resolved_image.save(output_path)

# Example usage
input_image_path = "input_image.jpg"  # Provide the path to your input low-resolution image
output_image_path = "super_resolved_image.jpg"  # Output path for the super-resolved image

# Perform super resolution
super_resolve_image(input_image_path, output_image_path)

print("Super resolution completed. Super-resolved image saved at", output_image_path)