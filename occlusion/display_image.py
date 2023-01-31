# -*- coding: utf-8 -*-
# @File    : display_image.py
# @Time    : 31/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm


import os
import sys
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)
from mrcnn import visualize
from mrcnn.visualize import display_images

dataset_dir = r'..\..\datasets\dataset_occluded'
annotation_dir = os.path.join(dataset_dir, 'annotations')
images_dir = os.path.join(dataset_dir, 'images')
lists_dir = os.path.join(dataset_dir, 'lists')

par_dir1 = "aeroplaneFGL1_BGL1"
image1 = "n02690373_16"
image1_path = os.path.join(images_dir, par_dir1, image1) + '.jpeg'
image1 = skimage.io.imread(image1_path)
par_dir2 = "bicycleFGL1_BGL2"
image2 = "n02834778_6158"
image2_path = os.path.join(images_dir, par_dir2, image2) + '.jpeg'
image2 = skimage.io.imread(image2_path)

display_images([image1, image2])
# plt.imshow(image)
# plt.show()
