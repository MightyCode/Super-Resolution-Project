import cv2 
import numpy as np
from typing import Callable, List

class Reducer:
    def __init__(self):
        self._max: Callable[[List[int]], int] = lambda array: max(array)
        self._mean: Callable[[List[int]], int] = lambda array: sum(array)//4

    def opencv_reduction(self, path: str, reduction_factor: float= 0.5):
        img=cv2.imread(path)

        return cv2.resize(path, None, fx = reduction_factor, fy = reduction_factor)
    
    def max_pooling_2D(self, path: str):
        return self._pooling_2D(path, function=self._max)

    def mean_pooling_2D(self, path: str):
        return self._pooling_2D(path, function=self._mean)
                   
    def _pooling_2D(self, path: str, function: Callable[[List[int]], int]):
        img=cv2.imread(path)

        width, height, channels = img.shape
        width, height = int(width/2), int(height/2)

        b_input_channel = img[:,:,0] # get blue channel
        g_input_channel = img[:,:,1] # get green channel
        r_input_channel = img[:,:,2] # get red channel

        result = np.zeros((width, height, channels))
    
        for i in range(width):
            for j in range(height):
                values = self._extract_square(r_input_channel, g_input_channel, b_input_channel, i, j)

                r = sum([rgb[0] for rgb in values])//4
                g = sum([rgb[1] for rgb in values])//4
                b = sum([rgb[2] for rgb in values])//4

                result[i][j][0] = b
                result[i][j][1] = g
                result[i][j][2] = r

        return result

    def _extract_square(self, r_channel, g_channel, b_channel, i: int, j: int):
        return [
            self._get_rgb_values_by_index(r_channel, g_channel, b_channel, i*2, j*2),
            self._get_rgb_values_by_index(r_channel, g_channel, b_channel, i*2+1, j*2),
            self._get_rgb_values_by_index(r_channel, g_channel, b_channel, i*2, j*2+1),
            self._get_rgb_values_by_index(r_channel, g_channel, b_channel, i*2+1, j*2+1)
        ]
    
    def _get_rgb_values_by_index(self, r_channel, g_channel, b_channel, i: int, j: int):
        return r_channel[i][j], g_channel[i][j], b_channel[i][j]
    

if __name__ == "__main__":
    reducer = Reducer()

    path: str = 'resources/alakazam.png'
    img = cv2.imread(path)

    res = reducer.mean_pooling_2D(path)
    saving_path: str = 'results/test.png'
    cv2.imwrite(saving_path, res)

    res = cv2.imread(saving_path)

    cv2.imshow('input', img)
    cv2.imshow('output', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()