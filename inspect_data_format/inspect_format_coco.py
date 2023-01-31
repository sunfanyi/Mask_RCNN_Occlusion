# -*- coding: utf-8 -*-
# @File    : inspect_format_coco.py
# @Time    : 30/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import os
import json
import time
import skimage.io
import skimage.draw
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


dataset_dir = 'D:\Desktop\FYP - Robust Surgical Tool Detection and Occlusion Handling using Deep Learning\datasets'
path_coco_captions = os.path.join(dataset_dir,
                                  'coco/annotations/captions_val2017.json')
path_coco_instances = os.path.join(dataset_dir,
                                   'coco/annotations/instances_minival2017.json')
captions = json.load(open(path_coco_captions))
# annotations = list(annotations.values())  # don't need the dict keys
instances = json.load(open(path_coco_instances))


# ================================= 1. coco() ==============================
# annotations = [a for a in balloon if a['regions']]
# dataset_dir = os.path.join(dataset_dir, 'balloon/val')

#
# def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR,
#               class_ids=None,
#               class_map=None, return_coco=False, auto_download=False):

coco = COCO(path_coco_instances)
image_dir = "{}/{}{}".format(dataset_dir, 'minival', '2017')

class_ids = sorted(coco.getCatIds())

# All images or a subset?
if class_ids:
    image_ids = []
    for id in class_ids:
        image_ids.extend(list(coco.getImgIds(catIds=[id])))
    # Remove duplicates
    image_ids = list(set(image_ids))
else:
    # All images
    image_ids = list(coco.imgs.keys())


def add_class(source, class_id, class_name, class_info):
    for info in class_info:
        if info['source'] == source and info["id"] == class_id:
            # source.class_id combination already available, skip
            return
    # Add the class
    class_info.append({
        "source": source,
        "id": class_id,
        "name": class_name,
    })
    return class_info


# Add classes
class_info = [{"source": "", "id": 0, "name": "BG"}]
for i in class_ids:
    class_info = add_class("coco", i, coco.loadCats(i)[0]["name"], class_info)


# ====================== 2. load_coco() =================
image_info = []

# Add images
for i in image_ids:
    image_info_dir = {
        "id": i,
        "source": 'coco',
        "path": os.path.join(image_dir, coco.imgs[i]['file_name']),
    }
    image_info_dir.update(width=coco.imgs[i]["width"],
                          height=coco.imgs[i]["height"],
                          annotations=coco.loadAnns(coco.getAnnIds(
                            imgIds=[i], catIds=class_ids, iscrowd=None)))
    image_info.append(image_info_dir)


# ========================= 3. prepare() =============================
# Build (or rebuild) everything else from the info dicts.
num_classes = len(class_info)
class_ids = np.arange(num_classes)
class_names = [c["name"] for c in class_info]
num_images = len(image_info)
_image_ids = np.arange(num_images)

# Mapping from source class and image IDs to internal IDs
class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                              for info, id in
                              zip(class_info, class_ids)}
image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                              for info, id in
                              zip(image_info, image_ids)}

# Map sources to class_ids they support
sources = list(set([i['source'] for i in class_info]))
source_class_ids = {}
# Loop over datasets
for source in sources:
    source_class_ids[source] = []
    # Find classes that belong to this dataset
    for i, info in enumerate(class_info):
        # Include BG class in all datasets
        if i == 0 or source == info['source']:
            source_class_ids[source].append(i)



# ========================= 4. load_mask() =======================
# take the first image as an example
image_id = 0
example_info = image_info[image_id]

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
    segm = ann['segmentation']
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
        rle = ann['segmentation']
    return rle

instance_masks = []
mask_class_ids = []
annotations = image_info[image_id]["annotations"]
for annotation in annotations:
    class_id = class_from_source_map["coco.{}".format(annotation['category_id'])]
    if class_id:  # if not 0 (background)
        # convert annotation to bitmap
        m = annToMask(annotation, example_info["height"],
                       example_info["width"])

        # Some objects are so small that they're less than 1 pixel area
        # and end up rounded out. Skip those objects.
        if m.max() < 1:
            continue
        # Is it a crowd? If so, use a negative class ID.
        if annotation['iscrowd']:
            # Use negative class ID for crowds
            class_id *= -1
            # For crowd masks, annToMask() sometimes returns a mask
            # smaller than the given dimensions. If so, resize it.
            if m.shape[0] != example_info["height"] or m.shape[1] != example_info["width"]:
                m = np.ones([example_info["height"], example_info["width"]], dtype=bool)
        instance_masks.append(m)
        mask_class_ids.append(class_id)
    break
# Pack instance masks into an array
if mask_class_ids:
    mask = np.stack(instance_masks, axis=2).astype(np.bool)
    mask_class_ids = np.array(class_ids, dtype=np.int32)

