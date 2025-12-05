# python program to perfrom morphological operations on a grayscale image
import cv2
import numpy as np
from matplotlib import pyplot as plt

def morphological_operations(image_path):
    # Read the image
    img = cv2.imread(image_path, 0)

    # Define a kernel
    kernel = np.ones((5,5), np.uint8)

    # Erosion
    erosion = cv2.erode(img, kernel, iterations = 1)

    # Dilation
    dilation = cv2.dilate(img, kernel, iterations = 1)

    # Morphological Gradient
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    # Top Hat
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    # Black Hat
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    # Plotting the results
    titles = ['Original Image', 'Erosion', 'Dilation', 'Gradient', 'Top Hat', 'Black Hat']
    images = [img, erosion, dilation, gradient, tophat, blackhat]

    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(2, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()

image_path = r'F:\Derain_code_and_github\input_images\signs.jpg'  # Replace with your image path
morphological_operations(image_path)