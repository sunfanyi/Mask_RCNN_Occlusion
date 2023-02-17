# -*- coding: utf-8 -*-
# @File    : plot_labelled.py
# @Time    : 12/02/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm


import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils


def annToMask(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m


def annToRLE(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann
    return rle



json_path = r"D:\Desktop\FYP - Robust Surgical Tool Detection and Occlusion Handling using Deep Learning\datasets\try\json\n02690373_16.json"
image_path = r"D:\Desktop\FYP - Robust Surgical Tool Detection and Occlusion Handling using Deep Learning\datasets\try\n02690373_16.JPEG"

anno = json.load(open(json_path))

height = anno['imageHeight']
width = anno['imageWidth']
mask = anno['shapes'][0]['points']
bitmap = annToMask([[item for sublist in mask for item in sublist]], height, width)
plt.imshow(bitmap.astype(np.uint8))
plt.show()
