import sys
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.signal import convolve2d
from scipy.ndimage import convolve, convolve1d

# Sobel filter in the Y direction
sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

d = np.array([[-2, -1, 0],
              [ -1, 1, 1],
              [ 0, 1, 2]])


e = np.array([[0.38, 0, 0],
              [ 0, 0.38, 0],
              [ 0, 0, 0.38]])

f = np.array([[0, -1, 0],
              [-1, 6, -1],
              [0, -1, 0]])


# Convolve the Sobel filter with itself
custom_kernel = convolve2d(sobel_y, sobel_y, mode='full')

image_path = 'stop.jpeg'  # Update this path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply the custom filter using OpenCV
# filtered_image = cv2.filter2D(image, -1, custom_kernel)
filtered_image = cv2.filter2D(image, -1, d)


# Display the original and filtered images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.show()