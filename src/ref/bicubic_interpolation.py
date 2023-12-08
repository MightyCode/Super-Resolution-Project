import numpy as np
import matplotlib.pyplot as mp
import cv2
import sys
import os.path


def BicubicInterpolation(img, scaleFactor):
    height, width, deep = img.shape
    newHeight = int(height * scaleFactor)
    newWidth = int(width * scaleFactor)

    newImg = np.zeros((newHeight, newWidth, deep))

    for line in range(newHeight):
        for column in range(newWidth):
            posX = line / scaleFactor
            posY = column / scaleFactor

            x = int(posX)
            y = int(posY)

            deltaX = posX - x
            deltaY = posY - y

            coeffX = [cubicFunction(deltaX + 1), cubicFunction(deltaX), cubicFunction(1 - deltaX), cubicFunction(2 - deltaX)]
            coeffY = [cubicFunction(deltaY + 1), cubicFunction(deltaY), cubicFunction(1 - deltaY), cubicFunction(2 - deltaY)]

            pixel_value = np.zeros(deep)
            for i in range(4):
                for j in range(4):
                    tmpX = min(max(x - 1 + j, 0), height - 1)
                    tmpY = min(max(y - 1 + i, 0), width - 1)

                    pixel_value += coeffX[j] * coeffY[i] * img[tmpX, tmpY]

            newImg[line, column] = pixel_value

    return newImg

def cubicFunction(t):
    if t > 2:
        return 0
    return t**3 - 2*t**2 + 1



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
    
    newImg = BicubicInterpolation(img, 2)
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