import os, sys

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent_dir)

if __name__ == "__main__":
    import numpy as np
    import cv2
    from metrics import Metric

    path1 = "resources/set14/baboon.png" if len(sys.argv) < 2 else sys.argv[1]
    image = cv2.imread(path1)

    inverted_image = cv2.bitwise_not(image)
    print(image[:1], inverted_image[:1])

    print(f"MSE : {Metric.MSE(image, inverted_image)}")
    print(f"PSCN : {Metric.PSCN(image, inverted_image)}")
    print(f"SSMH : {Metric.SSMH(image, inverted_image)}")

    
