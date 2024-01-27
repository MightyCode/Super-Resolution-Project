import streamlit as st
import numpy as np
import argparse
import torch
import os
import cv2
from PIL import Image
from src.utils.PytorchUtil import PytorchUtil as torchUtil
from torchvision import transforms
from src.models.InitModel import InitModel

common_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Function to preprocess the image for the model
def get_image():

    uploaded_img = st.file_uploader("Choose a file", type="png")
    return uploaded_img

# Function to make predictions
def predict(model, lr_data_numpy, device) -> np.ndarray:
    # Preprocess image by reversing the channels and applying the transform
    lr_data_torch = common_transform(lr_data_numpy).to(device)

    print(lr_data_torch.shape, type(lr_data_torch))
    pred_img_tensor = model(lr_data_torch)

    return torchUtil.tensor_to_numpy(pred_img_tensor)

""" 
Create an argument parser :
-p --path (str) mandatory : path to the model
-c --channel (str) mandatory : channel used by the model 
-u --upscale (int) : upscale factor
"""
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to the model", type=str)
    parser.add_argument("-c", "--channel", help="channel used by the model", type=str)
    parser.add_argument("-u", "--upscale", help="upscale factor", type=int)

    return parser

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    RESULT_PATH = "results/"

    PATH = "weights-upscale-residual-lpis-v.2" if args.path is None else args.path
    PATH = RESULT_PATH + PATH    

    CHANNELS = "bgr" if args.channel is None else args.channel
    UPSCALE_FACTOR = 2 if args.upscale is None else args.upscale

    CHANNEL_INTERPOLATIONS = {
        "b" : "bicubic",
        "g" : "bicubic",
        "r" : "bicubic",
        "d" : "nearest",
        "s" : "nearest",
    }

    (f"Usage : streamlit run app.py <model_name> <model_path>")
    print(f"Default is :", PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mod = InitModel.create_model(
        PATH, 
        {
            "learningRate": 0.001,
            "channel_positions" : CHANNELS,
            "channel_interpolations" : [CHANNEL_INTERPOLATIONS[c] for c in CHANNELS],
        }, 
        UPSCALE_FACTOR, device)

    # Streamlit app
    st.title("Image Super resolution")
    img = get_image()
    # Display the uploaded image
    if img is not None:
        img = Image.open(img)
        img = np.array(img)
        lr_data_numpy = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        st.image(torchUtil.numpy_to_image(lr_data_numpy), caption="Uploaded Image.", use_column_width=False)
            
        st.subheader(str(lr_data_numpy.shape))
        pred_img_numpy = predict(mod, lr_data_numpy, device)

        st.subheader("Result:")
        st.image(torchUtil.numpy_to_image(pred_img_numpy), caption="Super resolution Image.", use_column_width=False)

