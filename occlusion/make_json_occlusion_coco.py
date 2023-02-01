# -*- coding: utf-8 -*-
# @File    : make_json_occlusion_coco.py
# @Time    : 01/02/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# The raw annotation files are very slow to read (in npz). Therefore json files
# will be created and useful information will be written into it.

import os
import json
import numpy as np
import skimage.io

from pycocotools import mask
from pycococreatortools.pycococreatortools import binary_mask_to_polygon, \
    resize_binary_mask

dataset_dir = r'..\..\datasets\dataset_occluded'
annos_dir = os.path.join(dataset_dir, 'annotations')
images_dir = os.path.join(dataset_dir, 'images')
lists_dir = os.path.join(dataset_dir, 'lists')

cateogories = [{'id': 1, 'name': 'aeroplane'},
               {'id': 2, 'name': 'bicycle'},
               {'id': 3, 'name': 'boat'},
               {'id': 4, 'name': 'bottle'},
               {'id': 5, 'name': 'bus'},
               {'id': 6, 'name': 'car'},
               {'id': 7, 'name': 'chair'},
               {'id': 8, 'name': 'diningtable'},
               {'id': 9, 'name': 'motorbike'},
               {'id': 10, 'name': 'sofa'},
               {'id': 11, 'name': 'train'},
               {'id': 12, 'name': 'tvmonitor'},
               {'id': 13, 'name': 'occluder'}]
id_from_name_map = {info['name']: info['id']
                    for info in cateogories}


def add_image(image_info, source, image_id, path, **kwargs):
    info = {
        "id": image_id,
        "source": source,
        "path": path,
    }
    info.update(kwargs)
    image_info.append(info)


def create_image_info(image_id, file_name, par_dir, image_size):
                      # date_captured=datetime.datetime.utcnow().isoformat(' '),
                      # license_id=1, coco_url="", flickr_url=""):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "par_dir": par_dir,
            "width": image_size[0],
            "height": image_size[1],
            # "date_captured": date_captured,
            # "license": license_id,
            # "coco_url": coco_url,
            # "flickr_url": flickr_url
    }

    return image_info


def create_annotation_info(annotation_id, image_id, category_id, binary_mask,
                           image_size=None, tolerance=2, bbox=None):

    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask, image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bbox is None:
        bbox = mask.toBbox(binary_mask_encoded)

    is_crowd = 0
    segmentation = binary_mask_to_polygon(binary_mask, tolerance)
    if not segmentation:
        return None

    annotation_info = {
        "segmentation": segmentation,
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


def add_image_to_list(main_dict, file_name, par_dir, anno_id):
    image_id = file_name.split('.')[0]
    image_path = os.path.join(images_dir, par_dir, file_name)

    # get height and width from jpeg
    image = skimage.io.imread(image_path)
    image_size = image.shape[:2]

    image_info = create_image_info(image_id, file_name, par_dir, image_size)
    main_dict['images'].append(image_info)

    # get annotation information from npz
    npz_path = os.path.join(annos_dir, par_dir, image_id) + '.npz'
    npz_info = np.load(npz_path, allow_pickle=True)
    category_name = str(npz_info['category'])
    category_id = id_from_name_map[category_name]

    # bbox format is [y1, y2, x1, x2, img_h, img_w]
    # this is replaced to coco format with [y1, x1, y2, x2]
    bbox = npz_info['box'][:4].astype('float')
    bbox[[1, 2]] = bbox[[2, 1]]
    occluder_box = npz_info['occluder_box'][:, :4].astype('float')
    occluder_box[:, [1, 2]] = occluder_box[:, [2, 1]]

    mask = (npz_info['mask'] > 200)  # convert to boolean
    occluder_mask = npz_info['occluder_mask']

    annotation_info = create_annotation_info(anno_id, image_id, category_id,
                         mask, image_size=image_size, tolerance=2, bbox=bbox)
    main_dict['annotations'].append(annotation_info)
    anno_id += 1
    # for occluder_mask in occluder_masks:
    annotation_info = create_annotation_info(anno_id, image_id,
                     id_from_name_map['occluder'], occluder_mask,
                     image_size=image_size, tolerance=2, bbox=occluder_box)
    main_dict['annotations'].append(annotation_info)
    anno_id += 1
    return main_dict, anno_id


if __name__ == "__main__":
    count = 0
    main_dict = {'images': [], 'annotations': [], 'categories': cateogories}
    anno_id = 1
    for lst in os.listdir(lists_dir):
        # for each par_dir
        par_dir = lst.split('.')[0]
        with open(os.path.join(lists_dir, lst)) as file:
            file_names = file.readlines()
        for file_name in file_names:
            # for each image
            file_name = file_name.strip()

            main_dict, anno_id = add_image_to_list(main_dict, file_name,
                                                   par_dir, anno_id)

    with open("annotations_occlusion_coco_format.json", "w") as outfile:
        json.dump(main_dict, outfile)
