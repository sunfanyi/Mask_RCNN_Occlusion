# -*- coding: utf-8 -*-
# @File    : inspect_npz.py
# @Time    : 02/02/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

dataset_dir = '../../datasets/dataset_occluded'

file_name = 'aeroplaneFGL1_BGL1/n02690373_16'
annotation_path = os.path.join(dataset_dir, 'annotations', file_name) + '.npz'
image_path = os.path.join(dataset_dir, 'images', file_name) + '.jpeg'

annotation = np.load(annotation_path, allow_pickle=True)
image = skimage.io.imread(image_path)

# plt.figure()
# plt.imshow(image.astype(np.uint8))
# plt.axis('off')
# plt.show()
#
# print(annotation.files)
# print(annotation['source'])
# print(image.shape)
#
# # show original mask and saved mask (after compression)
# # mask = np.unpackbits(annotation['mask'])
# mask = (annotation['mask'] > 200)
# plt.figure()
# plt.title('occluded original mask')
# plt.axis('off')
# plt.imshow(mask.astype(np.uint8))
# plt.show()
