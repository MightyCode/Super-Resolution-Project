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

def upsale_video(video_path: str, model, video_out: str, show):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la vidéo.")
        exit()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(5)

    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, lr_data_numpy = cap.read()

        if not ret:
            break

        if show:
            cv2.imshow('frame', lr_data_numpy)


        lr_data_torch = common_transform(lr_data_numpy).to(device)

        pred_img_tensor = model(lr_data_torch.unsqueeze(0)).squeeze(0)

        pred_img_np = torchUtil.tensor_to_numpy(pred_img_tensor)

        output_image = (pred_img_np * 255.0).astype(np.uint8)

        if show:
            cv2.imshow('Sortie du modele', output_image)

        out.write(output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to the model", type=str, default="weights-upscale-residual-lpis-v.2")
    parser.add_argument("-c", "--channel", help="channel used by the model", type=str, default="bgr")
    parser.add_argument("-u", "--upscale", help="upscale factor", type=int, default=2)
    parser.add_argument("-v", "--video", help="upscale video", type=str, default="resources/video.mp4")
    parser.add_argument("-o", "--output", help="upscale video", type=str, default='results/video_out.mp4')
    parser.add_argument("-s", "--show", help="show video", type=bool, default=False)

    return parser


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    RESULT_PATH = "results/"

    PATH = args.path
    PATH = RESULT_PATH + PATH    

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
    
    model = InitModel.create_model(
        PATH, 
        {
            "learningRate": 0.001,
            "channel_positions" : CHANNELS,
            "channel_interpolations" : [CHANNEL_INTERPOLATIONS[c] for c in CHANNELS],
        }, 
        UPSCALE_FACTOR, device)

    upsale_video(args.video, model, args.output, args.show)