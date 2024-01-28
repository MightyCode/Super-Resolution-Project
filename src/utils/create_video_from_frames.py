import os
import sys
import cv2

def search_images(folder_path, start_frame, end_frame):
    # Ensure folder path ends with a slash
    folder_path = folder_path.rstrip(os.path.sep) + os.path.sep

    images = []
    for frame_number in range(start_frame, end_frame + 1):
        # Format the frame number with leading zeros
        frame_name = f"{frame_number:05d}.png"
        image_path = os.path.join(folder_path, frame_name)

        # Check if the image file exists
        if os.path.exists(image_path):
            images.append(image_path)
        else:
            print(f"Warning: Image not found - {image_path}")

    return images

""" 
Can be alreay an image
"""
def get_image(image_path):
    if type(image_path) == str:
        return cv2.imread(image_path)

    return image_path

def create_video(images, output_path, fps=10):
    if not images:
        print("No images to create a video.")
        return

    # Read the first image to get dimensions
    first_image = get_image(images[0])
    height, width, _ = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_path in images:
        frame = get_image(image_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video created at: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <folder_path> <start_frame> <end_frame> <output_video_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    start_frame = int(sys.argv[2])
    end_frame = int(sys.argv[3])
    output_video_path = sys.argv[4]

    images = search_images(folder_path, start_frame, end_frame)
    create_video(images, output_video_path)
