import test_waifu2x

# Set the paths for your input and output images
input_image = "input.jpg"
output_image = "output.jpg"

# Set the model to use (you can choose from 'noise', 'scale', or 'noise_scale')
model = 'scale'

# Specify the scale factor (e.g., 2x, 4x)
scale = 2

# Perform super-resolution
test_waifu2x.convert(model, scale, input_image, output_image)

print("Super-resolution complete. Output image saved as", output_image)
