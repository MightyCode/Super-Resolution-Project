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
def predict(mod, lr_data_numpy, device) -> np.ndarray:
    # Preprocess image by reversing the channels and applying the transform
    lr_data_torch = common_transform(lr_data_numpy).to(device)

    # lr_img_tensor = torchUtil.filter_data_to_img(lr_data_torch)

    pred_img_torch = mod(lr_data_torch.unsqueeze(0)).squeeze(0)

    pred_img_np = torchUtil.tensor_to_numpy(pred_img_torch)
    pred_img = torchUtil.numpy_to_image(pred_img_np)

    return pred_img

""" 
Create an argument parser :
-p --path (str) mandatory : path to the model
-c --channel (str) mandatory : channel used by the model 
-u --upscale (int) : upscale factor
"""
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to the model", type=str, default="weights-upscale-residual-lpips-v.2")
    parser.add_argument("-c", "--channel", help="channel used by the model", type=str, default="bgr")
    parser.add_argument("-u", "--upscale", help="upscale factor", type=int, default=2)

    return parser

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    RESULT_PATH = "results/"

    PATH = args.path  

    CHANNELS = args.channel
    UPSCALE_FACTOR = args.upscale

    CHANNEL_INTERPOLATIONS = {
        "b" : "bicubic",
        "g" : "bicubic",
        "r" : "bicubic",
        "d" : "nearest",
        "s" : "nearest",
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    mod = InitModel.create_model_static(
        PATH, PATH,
        {
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
        #save the image in the results folder
        img.save("results/image_tmp.png")
        img = np.array(img)
        lr_data_numpy = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        st.image(torchUtil.numpy_to_image(lr_data_numpy), caption="Uploaded Image.", use_column_width=False)

        lr_data_np = torchUtil.open_data("results/image_tmp.png")
        pred_img = predict(mod, lr_data_np, device)

        st.subheader("Result:")
        st.image(pred_img, caption="Super resolution Image.", use_column_width=False)

