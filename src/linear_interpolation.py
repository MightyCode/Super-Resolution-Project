import numpy as np
import matplotlib.pyplot as mp
import cv2

# export InterpolationLineaire

def InterpolationLineaire(img, scaleFactor):
    height, width, deep = img.shape
    newHeight = int(height * scaleFactor)
    newWidth = int(width * scaleFactor)
    
    newImg = np.zeros((newHeight, newWidth, deep))
    
    for line in range(newHeight):
        for column in range(newWidth):
            posX = line / scaleFactor
            posY = column / scaleFactor
            
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
            
            # for i in range(deep):  
            #     newValue[i] = (1 - deltaX) * (1 - deltaY) * img[x1, y1, i] + \
            #             (1 - deltaX) * deltaY * img[x1, y2, i] + \
            #             deltaX * (1 - deltaY) * img[x2, y1, i] + \
            #             deltaX * deltaY * img[x2, y2, i]
            
            newValue = (1 - deltaX) * (1 - deltaY) * img[x1, y1] + \
                    (1 - deltaX) * deltaY * img[x1, y2] + \
                    deltaX * (1 - deltaY) * img[x2, y1] + \
                    deltaX * deltaY * img[x2, y2]
                
            
            
            newImg[line, column] = newValue
            
    return newImg/255


if __name__ == "__main__":
    # img = cv2.imread('../E.png', cv2.IMREAD_GRAYSCALEheight)
    img = cv2.imread('../E.png')
    mp.imshow(img)
    mp.title("Image de faible résolution")
    mp.show()
    
    
    newImg = InterpolationLineaire(img, 2)
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
    cv2.imwrite('../E_resized.png', newImg)
    