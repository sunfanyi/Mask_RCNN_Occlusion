# -*- coding: utf-8 -*-
# @File    : Laplace_sharpen.py
# @Time    : 18/02/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm


import os
import sys
import cv2
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)
from mrcnn import visualize, utils
from mrcnn.model import log
# from mrcnn.visualize import display_images, draw_box
# from mrcnn.utils import minimize_mask, expand_mask
import occlusion

dataset_dir = '../../datasets/dataset_occluded'

dataset = occlusion.OcclusionDataset()
occlusion = dataset.load_occlusion(dataset_dir, "train", return_occlusion=True)
dataset.prepare()

target_id = "trainFGL1_BGL1/n02917067_4058"
for i in range(len(dataset.image_info)):
    if dataset.image_info[i]['id'] == target_id:
        target_idx = i
        break

matplotlib.use('tkagg')
# n = 1
# image_id = np.random.choice(dataset.image_ids, 1)[0]
image_id = 5
# image_id = target_idx

# Load image
# image_id = random.choice(dataset.image_ids)
image = dataset.load_image(image_id)

cv2.imshow('Original', image)

# ===================== Laplacian kernel ========================
# Define the Laplacian kernel
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

# Apply the Laplacian filter to the image
laplacian = cv2.filter2D(image, -1, laplacian_kernel)

# Add the Laplacian-filtered image to the original image to sharpen it
sharpened = image - laplacian

# Display the original and sharpened images
cv2.imshow('Laplacian kernel', sharpened)


# ========================== high-pass kernel ========================
hp_kernel = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])

# Apply the Laplacian filter to the image
sharpened = cv2.filter2D(image, -1, hp_kernel)

# Display the original and sharpened images
cv2.imshow('high-pass kernel', sharpened)


# ========================== Unsharp mask kernel ========================
unsharp_kernel = np.array([[-1 / 6, -2 / 3, -1 / 6],
                           [-2 / 3, 13 / 3, -2 / 3],
                           [-1 / 6, -2 / 3, -1 / 6]])

# Apply the Laplacian filter to the image
sharpened = cv2.filter2D(image, -1, unsharp_kernel)

# Display the original and sharpened images
cv2.imshow('Unsharp mask kernel', sharpened)



# ============== Laplacian kernel with splitting RGB ================
# Split the image into its R, G, and B components
b, g, r = cv2.split(image)

# Define the Laplacian kernel
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

# Apply the Laplacian filter to each color channel
lap_b = cv2.filter2D(b, -1, laplacian_kernel)
lap_g = cv2.filter2D(g, -1, laplacian_kernel)
lap_r = cv2.filter2D(r, -1, laplacian_kernel)

# Combine the filtered channels into a color image
lap_img = cv2.merge((lap_b, lap_g, lap_r))

sharpened = image - lap_img

# Display the original and sharpened images
cv2.imshow('Laplacian kernel with splitting RGB', sharpened)

# ============== high-pass kernel with splitting RGB ================
# Split the image into its R, G, and B components
b, g, r = cv2.split(image)

hp_kernel = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])

hp_b = cv2.filter2D(b, -1, hp_kernel)
hp_g = cv2.filter2D(g, -1, hp_kernel)
hp_r = cv2.filter2D(r, -1, hp_kernel)

sharpened = cv2.merge((hp_b, hp_g, hp_r))

cv2.imshow('high-pass kernel with splitting RGB', sharpened)


# ================== Unsharp mask kernel with splitting RGB ==================

b, g, r = cv2.split(image)

unsharp_kernel = np.array([[-1 / 6, -2 / 3, -1 / 6],
                           [-2 / 3, 13 / 3, -2 / 3],
                           [-1 / 6, -2 / 3, -1 / 6]])

us_b = cv2.filter2D(b, -1, unsharp_kernel)
us_g = cv2.filter2D(g, -1, unsharp_kernel)
us_r = cv2.filter2D(r, -1, unsharp_kernel)

sharpened = cv2.merge((us_b, us_g, us_r))

# Display the original and sharpened images
cv2.imshow('Unsharp mask kernel with splitting RGB', sharpened)


# ===================================================
cv2.waitKey(0)
cv2.destroyAllWindows()
