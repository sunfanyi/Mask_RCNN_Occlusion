# -*- coding: utf-8 -*-
# @File    : check_mask.py
# @Time    : 02/02/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm


import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

dataset_dir = os.path.abspath("")

file_name = 'aeroplaneFGL1_BGL1\\n02690373_16'
annotation_path = os.path.join(dataset_dir, 'annotations', file_name) + '.npz'
image_path = os.path.join(dataset_dir, 'images', file_name) + '.jpeg'

annotation = np.load(annotation_path, allow_pickle=True)
image = skimage.io.imread(image_path)

plt.figure()
plt.imshow(image.astype(np.uint8))
plt.axis('off')
plt.show()

print(annotation.files)
print(annotation['source'])
print(image.shape)

mask = (annotation['mask'] > 200)
plt.figure()
plt.title('occluded original mask')
plt.axis('off')
plt.imshow(mask.astype(np.uint8))
plt.show()

color = [255, 100, 100]
masked_image = image.copy()
alpha = 0.5
for c in range(3):
    masked_image[:, :, c] = np.where(mask == 1,
                                     # image[:, :, c] *
                                     (1 - alpha) + alpha * color[c] * 255,
                                     image[:, :, c])

plt.figure()
plt.title('masked_image')
plt.axis('off')
plt.imshow(masked_image.astype(np.uint8))
plt.show()
