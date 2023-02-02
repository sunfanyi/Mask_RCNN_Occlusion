# -*- coding: utf-8 -*-
# @File    : demo_mask2polygon.py
# @Time    : 01/02/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

"""
This line visulise the way to convert the mask in bitmap format to polygon
format for debugging purpose.
"""

import os
import sys
import numpy as np
import matplotlib

from skimage import measure
from pycocotools import mask as maskUtils
from pycococreatortools.pycococreatortools import binary_mask_to_polygon, \
    resize_binary_mask

import occlusion



ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)
from mrcnn import visualize, utils
from mrcnn.model import log
# from mrcnn.visualize import display_images, draw_box
# from mrcnn.utils import minimize_mask, expand_mask

dataset_dir = r'..\..\datasets\dataset_occluded'

dataset = occlusion.OcclusionDataset()
occlusion = dataset.load_occlusion(dataset_dir, "train", return_occlusion=True,
                                   mask_format='bitmap')
dataset.prepare()


def mask2polygon(mask):
    contours = measure.find_contours(mask, 0.5)
    res = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        res.append(segmentation)

    return res


def segm2mask(segm, height, width):
    # segm = ann['segmentation']
    rles = maskUtils.frPyObjects(segm, height, width)
    rle = maskUtils.merge(rles)
    mask = maskUtils.decode(rle)
    return mask


def mask2polygon(binary_mask, image_size=None, tolerance=2):

    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask,
                                         (image_size[1], image_size[0]))
    segmentation = binary_mask_to_polygon(binary_mask, tolerance)

    return segmentation


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


matplotlib.use('tkagg')

image_id = 0
image = dataset.load_image(image_id)
visualize.display_images([image], cols=1)

mask, class_ids = dataset.load_mask(image_id, mask_format='bitmap')
visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

# Compute Bounding box
bbox = utils.extract_bboxes(mask)

# Display image and additional stats
print("image_id ", image_id, dataset.image_reference(image_id))
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)

for m in mask:
    polygon = mask2polygon(m, image_size=None, tolerance=2)

# load image
# for image in image_info:
#     height = image['height']
#     width = image['width']
#     image_size = (height, width)
#
#     _image = skimage.io.imread(image['path'])
#     plt.figure()
#     plt.title("H x W={}x{}".format(height, width))
#     plt.axis('off')
#     plt.imshow(_image.astype(np.uint8))
#     plt.show()
#
#     ann = image['annotations'][0]
#     segm = np.array(ann['segmentation'])
    # mask = create_annotation_info(segm, image_size=image_size)
    #
    # m = annToMask(mask, height, width)
    # # mask = segm2mask(segm, height, width)
    # plt.figure()
    # plt.title('segmentation:' + image['annotations'][0]['category_name'])
    # plt.axis('off')
    # plt.imshow(m.astype(np.uint8))
    # plt.show()

    # segm = ann['occluder_mask']
    # occluder_mask = segm2mask(segm, height, width)
    # plt.figure()
    # plt.title('occluder_mask:' + image['annotations'][0]['category_name'])
    # plt.axis('off')
    # plt.imshow(occluder_mask.astype(np.uint8))
    # plt.show()

    # box = [int(i) for i in ann['bbox']]
    # _image_temp = draw_box(_image, box, np.array([255, 0, 0]))
    # plt.figure()
    # plt.title("bbox")
    # plt.axis('off')
    # plt.imshow(_image_temp.astype(np.uint8))
    # plt.show()

    # _image_temp = _image.copy()
    # for occluder_box in ann['occluder_box']:
    #     occluder_box = [int(i) for i in occluder_box]
    #     _image_temp = draw_box(_image, occluder_box, np.array([255, 0, 0]))
    # plt.figure()
    # plt.title("occluder_box")
    # plt.axis('off')
    # plt.imshow(_image_temp.astype(np.uint8))
    # plt.show()

    # mini_mask = minimize_mask([box], np.expand_dims(mask, axis=-1), (56, 56))
    # plt.figure()
    # plt.title('mini_mask:' + image['annotations'][0]['category_name'])
    # plt.axis('off')
    # plt.imshow(mini_mask[:, :, 0].astype(np.uint8))
    # plt.show()
    #
    # mask_expanded = expand_mask([box], mini_mask, (height, width))
    # plt.figure()
    # plt.title('expanded_mask:' + image['annotations'][0]['category_name'])
    # plt.axis('off')
    # plt.imshow(mask_expanded[:, :, 0].astype(np.uint8))
    # plt.show()
    # break