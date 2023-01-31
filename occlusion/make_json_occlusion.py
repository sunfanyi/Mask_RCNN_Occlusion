# -*- coding: utf-8 -*-
# @File    : make_json_occlusion.py
# @Time    : 31/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# The raw annotation files are very slow to read (in npz). Therefore json files
# will be created and useful information will be written into it.

# -*- coding: utf-8 -*-
# @File    : occluded.py
# @Time    : 30/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import os
import json
import numpy as np
import skimage.io

dataset_dir = r'..\..\datasets\dataset_occluded'
annos_dir = os.path.join(dataset_dir, 'annotations')
images_dir = os.path.join(dataset_dir, 'images')
lists_dir = os.path.join(dataset_dir, 'lists')
main_dict = {'images': [], 'categories': [{'id': 1, 'name': 'aeroplane'},
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
                                          {'id': 12, 'name': 'tvmonitor'}]}

id_from_name_map = {info['name']: info['id']
                    for info in main_dict['categories']}


def add_image(image_info, source, image_id, path, **kwargs):
    info = {
        "id": image_id,
        "source": source,
        "path": path,
    }
    info.update(kwargs)
    image_info.append(info)


for lst in os.listdir(lists_dir)[:1]:
    # for each par_dir
    par_dir = lst.split('.')[0]
    with open(os.path.join(lists_dir, lst)) as file:
        file_names = file.readlines()
    for file_name in file_names:
        # for each image
        file_name = file_name.strip()
        image_id = file_name.split('.')[0]
        image_path = os.path.join(images_dir, par_dir, file_name)

        # get height and width from jpeg
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]

        # get annotation information from npz
        npz_path = os.path.join(annos_dir, par_dir, image_id) + '.npz'
        npz_info = np.load(npz_path, allow_pickle=True)
        category_name = str(npz_info['category'])
        annotations = [{'segmentation': npz_info['mask'].tolist(),
                        'iscrowd': 0,
                        'image_id': image_id,
                        'bbox': npz_info['box'].tolist(),
                        'category_id': id_from_name_map[category_name],
                        'occluder_box': npz_info['occluder_box'].tolist(),
                        'occluder_mask': npz_info['occluder_mask'].tolist()}]

        add_image(main_dict['images'],
                  source='occlusion',
                  image_id=image_id,
                  path=image_path,
                  width=width,
                  height=height,
                  par_dir=par_dir,
                  annotations=annotations)
    break


# with open("annotations_occlusion.json", "w") as outfile:
#     json.dump(main_dict, outfile)
