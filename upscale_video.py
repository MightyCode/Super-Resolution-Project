from src.utils.PytorchUtil import PytorchUtil as torchUtil
from src.models.InitModel import InitModel
from src.utils.create_video_from_frames import create_video

from torchvision import transforms

import numpy as np
import argparse
import torch
import cv2

common_transform = transforms.Compose([
    transforms.ToTensor(),
])

def upsale_video(video_path: str, model, show):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la video.")
        exit()

    frames = []

    while True:
        ret, lr_data_numpy = cap.read()

        if not ret:
            break

        if show:
            cv2.imshow('frame', lr_data_numpy)

        lr_data_torch: torch.Tensor = common_transform(lr_data_numpy).to(device)
        
        pred_img_tensor: torch.Tensor = model(lr_data_torch.unsqueeze(0)).squeeze(0)

        pred_img_np: np.ndarray = torchUtil.tensor_to_numpy(pred_img_tensor)

        output = (pred_img_np * 255).astype(np.uint8)

        if show:
            cv2.imshow('super resolution', output)

        frames.append(output)
        print("a")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return frames

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--modelpath", help="path to the model", type=str, default="weights-upscale-residual-lpips-v.2")
    parser.add_argument("-c", "--channel", help="channel used by the model", type=str, default="bgr")
    parser.add_argument("-u", "--upscale", help="upscale factor", type=int, default=2)
    parser.add_argument("-v", "--video", help="upscale video", type=str, default="video.mp4")
    parser.add_argument("-o", "--output", help="upscale video", type=str, default='video_out.mp4')
    parser.add_argument("-s", "--show", help="show video", type=bool, default=False)

    return parser

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    RESOURCES_PATH = "resources/"
    RESULT_PATH = "results/" 

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

    exp = InitModel.create_model_static(
        args.modelpath, args.modelpath,
        {
            "learningRate": 0.001,
            "channel_positions" : CHANNELS,
            "channel_interpolations" : [CHANNEL_INTERPOLATIONS[c] for c in CHANNELS],
        }, 
        UPSCALE_FACTOR, device)

    with torch.no_grad():
        frames = upsale_video(RESOURCES_PATH + args.video, exp.net, args.show)

    create_video(frames, RESULT_PATH + args.output)