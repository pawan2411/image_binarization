import numpy as np
from PIL import Image, ImageDraw
from skimage import io, dtype_limits
import cv2

FINAL_IMAGE_COLOR_LIGHT = 255  # WHITE
FINAL_IMAGE_COLOR_DARK = 0  # BLACK
WINDOW_SIZE_FOR_AVG = 15  # DEFAULT SMOOTHNING WINDOW SIZE
K_PARAM_POSITIVE = 0.2  # K in adaptive filtering default value 0.2 (must be +ve)

"""
Name : main.py
Author : Pawan Kumar Rajpoot
Contect : pawan.rajpoot2411@gmail.com
Time    : 15th August 2021
Desc    : Solution for the question 2 for AI Lab | Verse Innovations Pvt. Ltd. Assignment
Reference   : http://www.unisoftimaging.com/download/AdaptiveDocumentImageBinarization1998.pdf
"""
def load_image(image_path):
    '''
    :param image_path: file path for input image
    :return: image instance plus details
    '''
    img = Image.open(image_path)
    image = img.convert('RGB')
    draw = ImageDraw.Draw(image)
    width, height = image.size[0], image.size[1]
    bw_matrix = io.imread(image_path, as_gray=True)
    return image, draw, width, height, bw_matrix


def draw_local(draw, width, height, binary_matrix, light_color, dark_color):
    '''
    :param draw: draw instance
    :param width: image width
    :param height: image height
    :param binary_matrix: matrix to write
    :param light_color: light color ASCII value
    :param dark_color: dark color ASCII value
    :return:
    '''
    for i in range(width):
        for j in range(height):
            if (binary_matrix[j, i] == 1):
                r, g, b = light_color, light_color, light_color
            else:
                r, g, b = dark_color, dark_color, dark_color
            draw.point((i, j), (r, g, b))


def apply_addaptive_filter(input_gray_matrix, window_size, binary_filtered_matrix_temp):
    '''
    :param input_gray_matrix: input image gray matrix
    :param window_size: window size for nXn filtering/averaging
    :param binary_filtered_matrix_temp: filter matrix
    :return: final filtered matrix for output image
    '''
    # dynamic range is usually difference of min max by four
    # refrence: https://www.thoughtco.com/range-rule-for-standard-deviation-3126231
    im_min, im_max = dtype_limits(input_gray_matrix)
    r = 0.25 * (im_max - im_min)

    mean = cv2.boxFilter(input_gray_matrix, cv2.CV_32F, (window_size, window_size),
                         borderType=cv2.BORDER_REPLICATE)
    square_mean = cv2.sqrBoxFilter(input_gray_matrix, cv2.CV_32F, (window_size, window_size),
                                   borderType=cv2.BORDER_REPLICATE)
    variance = square_mean - (mean * mean)
    standard_dev = np.sqrt(variance)
    # Equation 5 in http://www.unisoftimaging.com/download/AdaptiveDocumentImageBinarization1998.pdf
    threshold = mean * (1 + binary_filtered_matrix_temp * ((standard_dev / r) - 1))
    threshold = threshold.astype(input_gray_matrix.dtype)
    binary_filtered_matrix = (input_gray_matrix > threshold)
    binary_filtered_matrix = binary_filtered_matrix.astype(np.uint8)
    return binary_filtered_matrix


if __name__ == '__main__':
    image, draw, width, height, bw_matrix = load_image("sample.jpg")
    binary_filtered_matrix = apply_addaptive_filter(bw_matrix, WINDOW_SIZE_FOR_AVG, K_PARAM_POSITIVE)
    draw_local(draw, width, height, binary_filtered_matrix, FINAL_IMAGE_COLOR_LIGHT, FINAL_IMAGE_COLOR_DARK)
    image.save("result.jpg", "JPEG")
