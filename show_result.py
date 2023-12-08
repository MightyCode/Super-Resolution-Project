"""
Show result on histogram (save it) of results obtain from MEGA_TEST.py
"""
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import json


"""Create an argument parser that can handle
The dataset used, default is train, command is -d or --dataset
The upscale factor, default is 2, command is -u or --upscale
The possible methods, default is all, command is -m or --method

The result file used, default is "results/result1.json", command is -r or --result
The save file path, where the chart will be saved, default is "results/showresult.png", command is -p or --path
"""
def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="train", help="Dataset used, default is train")
    parser.add_argument("-u", "--upscale", default=2, help="Upscale factor, default is 2")
    parser.add_argument("-m", "--method", default="all", help="Possible methods, default is all")
    parser.add_argument("-r", "--result", default="results/result1.json", help="Result file used, default is results/result1.json")
    parser.add_argument("-p", "--path", default="results/showresult.png", help="Save file path, default is results/showresult.png")

    return parser

if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    dataset = args.dataset
    upscale = args.upscale
    method = args.method

    result = args.result
    path = args.path

    if not os.path.exists(result):
        print("Result file not found")
        exit()

    with open(result) as f:
        data = json.load(f)
    
    # Show an histogram with all the results
    data_number = len(data["entries"])

    # Create a list of all the results
    psnr_results = []
    ssim_results = []
    associated_model = []

    for entry in data["entries"]:
        if dataset != entry["dataset"] or upscale != str(entry["upscaleFactor"]):
            print("ho")
            continue

        psnr_results.append(entry["psnr"]["mean"])
        ssim_results.append(entry["ssim"]["mean"])

        associated_model.append(entry["model"]["name"] + "-" + entry["method"]["method"])

    # Create and show a figure with two subplots of histograms, one at right and one at left
    # Show in y-axis the value of psnr, show in x-axis the associated model name

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 40))
    fig.suptitle("Results obtained with dataset " + dataset + " and upscale factor " + str(upscale))

    # First subplot
    ax1.bar(associated_model, psnr_results)
    ax1.set_title("PSNR")
    ax1.set_ylabel("PSNR")
    ax1.set_xlabel("Model name")

    # Second subplot
    ax2.bar(associated_model, ssim_results)
    ax2.set_title("SSIM")
    ax2.set_ylabel("SSIM")
    ax2.set_xlabel("Model name")

    # logaritmic scale for y-axis
    ax1.set_yscale('log')
    ax2.set_yscale('log')

    # Rotate the x-axis labels
    plt.setp(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.setp(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')

    # Show the figure
    plt.show()

    # Save the figure
    fig.savefig(path)

    

