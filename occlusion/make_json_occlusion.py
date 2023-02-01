# -*- coding: utf-8 -*-
# @File    : make_json_occlusion.py
# @Time    : 31/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# The raw annotation files are very slow to read (in npz). Therefore json files
# will be created and useful information will be written into it.

import os
import json
import numpy as np
import skimage.io
from skimage import measure

from pycococreatortools.pycococreatortools import binary_mask_to_polygon

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
               {'id': 12, 'name': 'tvmonitor'}]
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


def add_image_to_list(image_info, file_name, par_dir):
    image_id = file_name.split('.')[0]
    image_path = os.path.join(images_dir, par_dir, file_name)

    # get height and width from jpeg
    image = skimage.io.imread(image_path)
    height, width = image.shape[:2]

    # get annotation information from npz
    npz_path = os.path.join(annos_dir, par_dir, image_id) + '.npz'
    npz_info = np.load(npz_path, allow_pickle=True)
    category_name = str(npz_info['category'])

    # bbox format is [y1, y2, x1, x2, img_h, img_w]
    # this is replaced to coco format with [y1, x1, y2, x2]
    [y1, y2, x1, x2] = npz_info['box'][:4].astype('float')
    bbox = [y1, x1, y2, x2]
    occluder_box = npz_info['occluder_box'][:, :4].astype('float')
    occluder_box[:, [1, 2]] = occluder_box[:, [2, 1]]

    # convert mask to polygon format to save memory
    mask = (npz_info['mask'] > 200)  # convert to boolean
    mask = binary_mask_to_polygon(mask, tolerance=2)
    occluder_mask = npz_info['occluder_mask']
    occluder_mask = binary_mask_to_polygon(occluder_mask, tolerance=2)

    annotations = [{'segmentation': mask,
                    'iscrowd': 0,
                    'image_id': image_id,
                    'bbox': bbox,
                    'category_id': id_from_name_map[category_name],
                    'category_name': category_name,
                    'occluder_box': occluder_box.tolist(),
                    'occluder_mask': occluder_mask}]

    add_image(image_info,
              source='occlusion',
              image_id=image_id,
              path=image_path,
              width=width,
              height=height,
              par_dir=par_dir,
              annotations=annotations)
    return image_info


if __name__ == "__main__":
    count = 0
    main_dict = {'images': [], 'categories': cateogories}
    for lst in os.listdir(lists_dir):
        # for each par_dir
        par_dir = lst.split('.')[0]
        with open(os.path.join(lists_dir, lst)) as file:
            file_names = file.readlines()
        for file_name in file_names:
            # for each image
            file_name = file_name.strip()

            main_dict['images'] = add_image_to_list(main_dict['images'],
                                                    file_name, par_dir)
    # with open("annotations_occlusion.json", "w") as outfile:
    #     json.dump(main_dict, outfile)
