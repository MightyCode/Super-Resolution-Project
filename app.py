import sys
import streamlit as st
import src.nntools as nt
import numpy as np
from torchvision import transforms
from PIL import Image
from src.InitModel import InitModel


common_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Function to preprocess the image for the model
def get_image():
    uploaded_img = st.file_uploader("Choose a file", type="png")
    return uploaded_img

# Function to make predictions
def predict(m, img):
    # Preprocess image by reversing the channels and applying the transform
    img = common_transform(img)
    new_img = m(img)
    new_img = new_img.squeeze(0)
    new_img = new_img.permute(1, 2, 0)
    new_img = new_img.detach().numpy()
    return new_img


if __name__ == "__main__":

    NAME = "superresol-upscale-old"
    PATH = "results/superresol-upscale2/"

    if len(sys.argv) == 3:
        NAME = sys.argv[1]
        PATH = sys.argv[2]
    else:
        print(f"Usage : streamlit run app.py <model_name> <model_path>")
        print(f"Default is :", NAME, PATH)


    UPSCALE_FACTOR = 2
    mod = InitModel.create_model(NAME, PATH, {"learningRate": 0.001}, UPSCALE_FACTOR, 'cpu')
    # Streamlit app
    st.title("Image Super resolution")
    img = get_image()
    # Display the uploaded image
    if img is not None:
        img = Image.open(img)
        img = np.array(img)
        print(img.shape)
        st.image(img, caption="Uploaded Image.", use_column_width=False)
            
        pred_img = predict(mod, img)

        st.subheader("Result:")
        st.image(pred_img, caption="Super resolution Image.", use_column_width=False)

