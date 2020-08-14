"""
Denoise Problem
(Due date: Nov. 25, 11:59 P.M., 2019)
The goal of this task is to denoise image using median filter.

Do NOT modify the code provided to you.
Do NOT import ANY library or API besides what has been listed.
Hint: 
Please complete all the functions that are labeled with '#to do'. 
You are suggested to use utils.zero_pad.
"""

import utils
import numpy as np
import json


def median_filter(img):
    """
    Implement median filter on the given image.
    Steps:
    (1) Pad the image with zero to ensure that the output is of the same size as the input image.
    (2) Calculate the filtered image.
    Arg: Input image. 
    Return: Filtered image.
    """
    # TODO: implement this function.

    padded_image = utils.zero_pad(img, 1, 1)  # Image with zero padding
    padded_image_h = len(padded_image)  # Height of zero padded image
    padded_image_w = len(padded_image[0])  # Width of zero padded image
    output = img

    for x in range(padded_image_h - (1 * 2)):
        for y in range(padded_image_w - (1 * 2)):
            patch = padded_image[x: x + 3, y: y + 3]  # 3 * 3 patch
            patch_median = np.median(patch)  # calculate median for 3 * 3 patch
            output[x][y] = patch_median  # implementing filtered image
    return output


def mse(img1, img2):
    """
    Calculate mean square error of two images.
    Arg: Two images to be compared.
    Return: Mean square error.
    """
    # TODO: implement this function.
    for i in range(len(img1)):
        for j in range(len(img2)):
            mean_square_error = np.square(np.subtract(img1, img2)).mean()  # calculating mean square error
    return mean_square_error


if __name__ == "__main__":
    img = utils.read_image('lenna-noise.png')
    gt = utils.read_image('lenna-denoise.png')

    result = median_filter(img)
    error = mse(gt, result)

    with open('results/task2.json', "w") as file:
        json.dump(error, file)
    utils.write_image(result, 'results/task2_result.jpg')
