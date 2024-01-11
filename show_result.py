"""
Show result on histogram (save it) of results obtain from MEGA_TEST.py
"""
import argparse
import os
import matplotlib.pyplot as plt
import json
import matplotlib.patches as mpatches


"""Create an argument parser that can handle
The dataset used, default is train, command is -d or --dataset
The upscale factor, default is 2, command is -u or --upscale
The possible methods, default is all, command is -m or --method
The result file used, default is "results/result1.json", command is -r or --result
The save file path, where the chart will be saved, default is "results/showresult.png", command is -p or --path
The order option takes no arguments, it is used to order the results by psnr or ssim, command is -o or --order
"""
def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="train", help="Dataset used, default is train")
    parser.add_argument("-u", "--upscale", default=2, help="Upscale factor, default is 2")
    parser.add_argument("-m", "--method", default="all", help="Possible methods, default is all")
    parser.add_argument("-p", "--path", default="results/result1.json", help="Result file used, default is results/result1.json")
    parser.add_argument("-r", "--result", default="results/showresult.png", help="Save file path, default is results/showresult.png")
    parser.add_argument("-o", "--order", action="store_true", help="Order the results")

    return parser


def sort_label_metric(label_array, metric_array):
    # Combine data into a list of tuples (model, PSNR, SSIM)
    combined_data = list(zip(label_array, metric_array))

    # Sort the combined data by PSNR values first and then by SSIM values
    combined_data.sort(key=lambda x: (x[1]))  # Sorting by PSNR and then by SSIM

    # Unpack sorted data
    label_array, metric_array = zip(*combined_data)

    return label_array, metric_array


def give_order(metric_array):
    # Give the position in classement, result array is same size as metric_array
    result_array = []

    # Sort the metric array
    sorted_metric_array = sorted(metric_array)

    # For each element in metric_array, find the position in sorted_metric_array
    for element in metric_array:
        result_array.append(sorted_metric_array.index(element))


def get_color_name_bar(associated_model, name_separation="-image", interpolation_methods=[]):
    name_color = []
    bar_color = []

    # if the associated_model contains -image, color it with redish color, otherwise with blueish color
    for model in associated_model:
        if name_separation in model:
            name_color.append("purple")
        else:
            name_color.append("green")

        found = False
        for alternative in interpolation_methods:
            if alternative in model:
                bar_color.append("red")
                found = True
                break

        if not found:
            bar_color.append("blue")

    return name_color, bar_color


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    dataset = args.dataset
    upscale = args.upscale
    method = args.method

    path = args.path
    result = args.result

    if not os.path.exists(path):
        print("Result file not found")
        exit()

    with open(path) as f:
        data = json.load(f)
    
    # Show an histogram with all the results
    data_number = len(data["entries"])

    INTERPOLATION_METHODS = ["bicubic", "nearest", "area", "lanczos", "bilinear"]

    # Create a list of all the results
    psnr_results = []
    ssim_results = []
    associated_model_psnr = []
    associated_model_ssim = []

    for entry in data["entries"]:
        if dataset != entry["dataset"] or upscale != str(entry["upscaleFactor"]):
            continue

        psnr_results.append(entry["psnr"]["mean"])
        ssim_results.append(entry["ssim"]["mean"])

        associated_model_psnr.append(entry["model"]["name"] + "-" + entry["method"]["method"])
        associated_model_ssim.append(associated_model_psnr[-1])

    if args.order:
        associated_model_psnr, psnr_results = sort_label_metric(associated_model_psnr, psnr_results)

    name_color, bar_color = get_color_name_bar(associated_model_psnr, interpolation_methods=INTERPOLATION_METHODS)

    # Create and show a figure with two subplots of histograms, one at right and one at left
    # Show in y-axis the value of psnr, show in x-axis the associated model name

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    fig.suptitle("Results obtained with dataset " + dataset + " and upscale factor " + str(upscale), 
                 fontsize=14, fontweight='bold')

    # First subplot, color the bar with random colors
    bars = ax1.bar(associated_model_psnr, psnr_results, color=bar_color)

    for label, color in zip(ax1.get_xticklabels(), name_color):
        label.set_color(color)
    
    ax1.set_title("PSNR")
    ax1.set_ylabel("PSNR")
    ax1.set_xlabel("Model names")

    if args.order:
        associated_model_ssim, ssim_results = sort_label_metric(associated_model_ssim, ssim_results)

    name_color, bar_color = get_color_name_bar(associated_model_ssim, interpolation_methods=INTERPOLATION_METHODS)

    # Second subplot
    ax2.bar(associated_model_ssim, ssim_results, color=bar_color)
    ax2.set_title("SSIM")
    ax2.set_ylabel("SSIM")
    ax2.set_xlabel("Model names")


    for label, color in zip(ax2.get_xticklabels(), name_color):
        label.set_color(color)

    # logaritmic scale for y-axis
    ax1.set_yscale('log')
    ax2.set_yscale('log')

    # Rotate the x-axis labels
    plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')

    blue_patch = mpatches.Patch(color='blue', label='Our Models')
    red_patch = mpatches.Patch(color='red', label='Interpolation Methods')

    # Adding legend
    ax1.legend(handles=[blue_patch, red_patch])
    ax2.legend(handles=[blue_patch, red_patch])

    # Show the figure
    plt.show()

    # Save the figure
    fig.savefig(result)

    


