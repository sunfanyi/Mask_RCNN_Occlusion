# -*- coding: utf-8 -*-
# @File    : temp_mask2polygon.py
# @Time    : 31/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import os
import json
import numpy as np
import skimage.io
from skimage import measure

# dataset_dir = r'..\..\datasets\dataset_occluded'
# annos_dir = os.path.join(dataset_dir, 'annotations')
# images_dir = os.path.join(dataset_dir, 'images')
# lists_dir = os.path.join(dataset_dir, 'lists')
#
# cateogories = [{'id': 1, 'name': 'aeroplane'},
#                {'id': 2, 'name': 'bicycle'},
#                {'id': 3, 'name': 'boat'},
#                {'id': 4, 'name': 'bottle'},
#                {'id': 5, 'name': 'bus'},
#                {'id': 6, 'name': 'car'},
#                {'id': 7, 'name': 'chair'},
#                {'id': 8, 'name': 'diningtable'},
#                {'id': 9, 'name': 'motorbike'},
#                {'id': 10, 'name': 'sofa'},
#                {'id': 11, 'name': 'train'},
#                {'id': 12, 'name': 'tvmonitor'}]
# id_from_name_map = {info['name']: info['id']
#                     for info in cateogories}
#
#
# def add_image(image_info, source, image_id, path, **kwargs):
#     info = {
#         "id": image_id,
#         "source": source,
#         "path": path,
#     }
#     info.update(kwargs)
#     image_info.append(info)
#
#


#
# def add_image_to_list(image_info, file_name, par_dir):
#     image_id = file_name.split('.')[0]
#     image_path = os.path.join(images_dir, par_dir, file_name)
#
#     # get height and width from jpeg
#     image = skimage.io.imread(image_path)
#     height, width = image.shape[:2]
#
#     # get annotation information from npz
#     npz_path = os.path.join(annos_dir, par_dir, image_id) + '.npz'
#     npz_info = np.load(npz_path, allow_pickle=True)
#     category_name = str(npz_info['category'])
#
#     # bbox format is [y1, y2, x1, x2, img_h, img_w]
#     # this is replaced to coco format with [y1, x1, y2, x2]
#     [y1, y2, x1, x2] = npz_info['box'][:4].astype('float')
#     bbox = [y1, x1, y2, x2]
#     occluder_box = npz_info['occluder_box'][:, :4].astype('float')
#     occluder_box[:, [1, 2]] = occluder_box[:, [2, 1]]
#
#     # convert mask to polygon format to save memory
#     mask = (npz_info['mask'] > 200)  # convert to boolean
#     print(np.array(mask))
#     mask = mask2polygon(np.array(mask))
#     occluder_mask = mask2polygon(npz_info['occluder_mask'])
#
#     annotations = [{'segmentation': mask,
#                     'iscrowd': 0,
#                     'image_id': image_id,
#                     'bbox': bbox,
#                     'category_id': id_from_name_map[category_name],
#                     'category_name': category_name,
#                     'occluder_box': occluder_box.tolist(),
#                     'occluder_mask': occluder_mask}]
#
#     add_image(image_info,
#               source='occlusion',
#               image_id=image_id,
#               path=image_path,
#               width=width,
#               height=height,
#               par_dir=par_dir,
#               annotations=annotations)
#     return image_info
#
#
# if __name__ == "__main__":
#     count = 0
#     main_dict = {'images': [], 'categories': cateogories}
#     for lst in os.listdir(lists_dir):
#         # for each par_dir
#         par_dir = lst.split('.')[0]
#         with open(os.path.join(lists_dir, lst)) as file:
#             file_names = file.readlines()
#         for file_name in file_names:
#             # for each image
#             file_name = file_name.strip()
#
#             main_dict['images'] = add_image_to_list(main_dict['images'],
#                                                     file_name, par_dir)
#             break
#         break
#     # with open("annotations_occlusion.json", "w") as outfile:
#     #     json.dump(main_dict, outfile)
# def mask2polygon(mask):
#     contours = measure.find_contours(mask, 0.5)
#     res = []
#     for contour in contours:
#         contour = np.flip(contour, axis=1)
#         segmentation = contour.ravel().tolist()
#         res.append(segmentation)
#
#     return res

image_path = r"D:\Desktop\FYP - Robust Surgical Tool Detection and Occlusion Handling using Deep Learning\datasets\dataset_occluded\images\aeroplaneFGL1_BGL1\n02690373_10111.JPEG"
npz_path = r"D:\Desktop\FYP - Robust Surgical Tool Detection and Occlusion Handling using Deep Learning\datasets\dataset_occluded\annotations\aeroplaneFGL1_BGL1\n02690373_10111.npz"
npz_info = np.load(npz_path, allow_pickle=True)
# get height and width from jpeg
image = skimage.io.imread(image_path)
height, width = image.shape[:2]
mask = npz_info['mask']
mask = (mask > 200).astype(np.uint8)  # convert to boolean

import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
from pycococreatortools.pycococreatortools import binary_mask_to_polygon

polygon = binary_mask_to_polygon(mask, tolerance=2)


def segm2mask(segm, height, width):
    # segm = ann['segmentation']
    rles = maskUtils.frPyObjects(segm, height, width)
    rle = maskUtils.merge(rles)
    mask = maskUtils.decode(rle)
    return mask

mask_saved = segm2mask(polygon, height, width)

plt.figure()
plt.title('mask_saved')
plt.axis('off')
plt.imshow(mask_saved.astype(np.uint8))
plt.show()

plt.figure()
plt.title('mask_original')
plt.axis('off')
plt.imshow(mask.astype(np.uint8))
plt.show()
