import numpy as np
import matplotlib.pyplot as mp
import cv2
import sys
import os.path


def BilinearInterpolation(img, scaleFactor):
    height, width, deep = img.shape
    newHeight = int(height * scaleFactor)
    newWidth = int(width * scaleFactor)

    newImg = np.zeros((newHeight, newWidth, deep))

    for line in range(newHeight):
        for column in range(newWidth):
            posX = line/scaleFactor
            posY = column/scaleFactor

            x1 = int(posX)
            x2 = x1 + 1
            y1 = int(posY)
            y2 = y1 + 1

            deltaX = posX - x1
            deltaY = posY - y1
            
            if x2 >= height:
                x2 = height - 1
            if y2 >= width:
                y2 = width - 1
            
            newValue = np.zeros(deep)

            valuePixelTopLeft = img[x1, y1] * (1 - deltaX) * (1 - deltaY)
            valuePixelTopRight = img[x1, y2] * (1 - deltaX) * deltaY
            valuePixelBottomLeft = img[x2, y1] * deltaX * (1 - deltaY)
            valuePixelBottomRight = img[x2, y2] * deltaX * deltaY

            newValue = valuePixelTopLeft + valuePixelTopRight + valuePixelBottomLeft + valuePixelBottomRight

            newImg[line, column] = newValue

    return newImg/255



if __name__ == "__main__":

    imgPath = '../datasets/dataset/train/high_res/0.png'

    if len(sys.argv) > 1:
        imgPath = sys.argv[1]
        if not os.path.isfile(imgPath):
            print("Error: file not found")
            sys.exit(1)

    if not os.path.isfile(imgPath):
        print("Error: No file specified: use 'python bilinear_interpolation.py <path_to_file>'")
        sys.exit(1)

    img = cv2.imread(imgPath)
    
    toPrint = False

    if toPrint:
        mp.imshow(img)
        mp.title("Image de faible résolution")
        mp.show()
    
    newImg = BilinearInterpolation(img, 2)
    if toPrint:
        mp.imshow(newImg)
        mp.title("Image de haute résolution")
        mp.show()

        fig = mp.figure()
        ax1 = fig.add_subplot(121)
        ax1.imshow(img)
        ax1.set_title("Image de faible résolution")
        ax2 = fig.add_subplot(122)
        ax2.imshow(newImg)
        ax2.set_title("Image de haute résolution")
        mp.show()
    
    newImg = newImg * 255
    print("Saving resised image as resized.png")
    cv2.imwrite('../resources/resized.png', newImg)
    
    newImg = newImg * 255