import numpy as np
import matplotlib.pyplot as plt
from test_convolution import convolution


def sobel_edge_detection(image, filter, convert_to_degree=False, verbose=False):
    
    new_image_x = convolution(image, filter, verbose)
    new_image_y = convolution(image, np.flip(filter.T, axis=0), verbose)

    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()

    gradient_direction = np.arctan2(new_image_y, new_image_x)

    if convert_to_degree:
        gradient_direction = np.rad2deg(gradient_direction)
        gradient_direction += 180

    return gradient_magnitude, gradient_direction