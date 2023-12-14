import streamlit as st
import src.nntools as nt
import src.UpscaleNN
import src.rdn
import numpy as np
from torchvision import transforms
from PIL import Image

# Load the pre-trained VGG model
un = src.UpscaleNN.UpscaleNN()
model = nt.Model(un, output_dir="results/smallbatchexperiment-upscale", device="cpu")
common_transform = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

# Function to preprocess the image for the model
def get_image():
    uploaded_img = st.file_uploader("Choose a file", type="png")
    return uploaded_img

# Function to make predictions
def predict(img):
    # Preprocess image by reversing the channels and applying the transform
    img = common_transform(img)
    new_img = model(img)
    new_img = new_img.squeeze(0)
    new_img = new_img.permute(1, 2, 0)
    new_img = new_img.detach().numpy()
    return new_img


if __name__ == "__main__":
    # Streamlit app
    st.title("Image Super resolution")
    img = get_image()
    # Display the uploaded image
    if img is not None:
        img = Image.open(img)
        img = np.array(img)
        print(img.shape)
        st.image(img, caption="Uploaded Image.", use_column_width=False)
            
        pred_img = predict(img)

        st.subheader("Result:")
        st.image(pred_img, caption="Super resolution Image.", use_column_width=False)

