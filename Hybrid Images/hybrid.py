import sys
import cv2
import numpy as np
import math
from scipy.signal import convolve2d
from scipy.ndimage import convolve, convolve1d

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    def compute_sum_centered_at(x, y, c):
        result = 0.0

        for u in range(-center_m, center_m + 1):
            for v in range(-center_n, center_n + 1):
                product = 0.0

                img_x = x + u
                img_y = y + v
                if 0 <= img_x < img_height and 0 <= img_y < img_width:
                    if channel == 1:
                        product = kernel[u + center_m, v + center_n] * img[img_x, img_y]
                    else:
                        product = kernel[u + center_m, v + center_n] * img[img_x, img_y, c]
                
                result += product

        return result

    m, n = kernel.shape
    center_m = m // 2
    center_n = n // 2

    img_shape = img.shape
    img_height = img_shape[0]
    img_width = img_shape[1]
    channel = 1 if len(img_shape) == 2 else img_shape[-1]
    
    new_image = np.zeros((img_height, img_width)) if channel == 1 else \
                np.zeros((img_height, img_width, channel))

    for x in range(img_height):
        for y in range(img_width):
            for c in range(channel):
                total = compute_sum_centered_at(x, y, c)
                
                if channel == 1:
                    new_image[x, y] = total
                else:
                    new_image[x, y, c] = total

    return new_image
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    m, n = kernel.shape
    flipped_kernel = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            flipped_kernel[i, j] = kernel[m - 1 - i][n - 1 - j]

    # placeholder = []
    # for i in range(m):
    #     for j in range(n):
    #         placeholder.append(kernel[i, j])

    # for i in range(m):
    #     for j in range(n):
    #         flipped_kernel[i, j] = placeholder.pop()

    return cross_correlation_2d(img, flipped_kernel)
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    gaussian_kernel = np.zeros((height, width))
    center_x = height // 2
    center_y = width // 2
    total_sum = 0.0

    for i in range(height):
        for j in range(width):
            x = i - center_x
            y = j - center_y
            
            left_side = 1 / (2 * math.pi * pow(sigma, 2))

            exponent = -((pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2)))
            right_side = math.exp(exponent)

            result = left_side * right_side
            gaussian_kernel[i, j] = result
            
            total_sum += result            
    
    gaussian_kernel /= total_sum
    return gaussian_kernel
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    return img - low_pass(img, sigma, size)
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

