# -*- coding: utf-8 -*-
# @File    : make_json_occlusion_coco.py
# @Time    : 17/02/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import os
import json
import numpy as np

from pycocotools import mask
from pycococreatortools.pycococreatortools import resize_binary_mask


def annToMask(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = mask.decode(rle)
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
        rles = mask.frPyObjects(segm, height, width)
        rle = mask.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = mask.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann
    return rle


def create_image_info(image_id, file_name, image_size):
    # date_captured=datetime.datetime.utcnow().isoformat(' '),
    # license_id=1, coco_url="", flickr_url=""):

    image_info = {
        "id": image_id,
        "file_name": file_name,
        # "par_dir": par_dir,
        "width": image_size[1],
        "height": image_size[0],
        # "date_captured": date_captured,
        # "license": license_id,
        # "coco_url": coco_url,
        # "flickr_url": flickr_url
    }

    return image_info


def create_annotation_info(annotation_id, image_id, category_id, polygon_mask,
                           binary_mask,
                           image_size=None, bbox=None):
    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask,
                                         (image_size[1], image_size[0]))

    binary_mask_encoded = mask.encode(
        np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bbox is None:
        bbox = mask.toBbox(binary_mask_encoded)

    is_crowd = 0
    m = polygon_mask

    annotation_info = {
        "segmentation": m,
        "area": area.tolist(),
        "iscrowd": is_crowd,
        "image_id": image_id,
        "bbox": bbox.tolist(),
        "category_id": category_id,
        "id": annotation_id,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    }

    return annotation_info


def add_image_to_list(main_dict, file_name, par_name, anno_id):
    image_id = file_name.split('.')[0]

    json_path = os.path.join(annos_dir, image_id) + '.json'
    json_info = json.load(open(json_path))

    image_size = [json_info['imageHeight'], json_info['imageWidth']]

    image_info = create_image_info(image_id, file_name, image_size)
    main_dict['images'].append(image_info)

    for anno in json_info['shapes']:  # for each mask
        category_name = anno['label']
        category_id = id_from_name_map[category_name]

        mask = anno['points']
        polygon_mask = [[item for sublist in mask for item in sublist]]
        binary_mask = annToMask(polygon_mask, image_size[0], image_size[1])

        annotation_info = create_annotation_info(anno_id, image_id, category_id,
                                                 polygon_mask, binary_mask,
                                                 image_size=image_size,
                                                 bbox=None)

        main_dict['annotations'].append(annotation_info)
        anno_id += 1

    return anno_id


dataset_dir = '../../datasets/dataset_occluded'
# dataset_dir = '../check_mask'

cateogories = [{'id': 1, 'name': 'aeroplane'},
               {'id': 4, 'name': 'bottle'},
               {'id': 5, 'name': 'bus'},
               {'id': 6, 'name': 'car'},
               {'id': 11, 'name': 'train'},
               {'id': 13, 'name': 'occluder'}]
id_from_name_map = {info['name']: info['id']
                    for info in cateogories}

annos_dir = os.path.join(dataset_dir, 'jsons_my_anno')
images_dir = os.path.join(dataset_dir, 'images')

train_dict = {'images': [], 'annotations': [],
              'categories': cateogories}
val_dict = {'images': [], 'annotations': [], 'categories': cateogories}

train_val_ratio = 4  # train:val ~= 20:1
anno_id = 1

for par_name in os.listdir(annos_dir):
    for image in os.listdir(os.path.join(annos_dir, par_name)):
        image_id = par_name + '/' + image
        file_name = image_id.replace('json', 'JPEG')
        anno_id = add_image_to_list(train_dict, file_name, par_name, anno_id)

