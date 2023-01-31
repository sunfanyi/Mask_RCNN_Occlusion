# -*- coding: utf-8 -*-
# @File    : display_image.py
# @Time    : 31/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm


import os
import sys
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
from mrcnn.utils import minimize_mask, expand_mask

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)
# from mrcnn import visualize
from mrcnn.visualize import display_images, draw_box

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


for lst in [os.listdir(lists_dir)[0], os.listdir(lists_dir)[10]]:
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

        # bbox format is [y1, y2, x1, x2, img_h, img_w]
        # this is replaced to coco format with [y1, x1, y2, x2]
        [y1, y2, x1, x2] = npz_info['box'][:4]
        bbox = [y1, x1, y2, x2]
        occluder_box = npz_info['occluder_box'][:, :4]
        occluder_box[:, [1, 2]] = occluder_box[:, [2, 1]]

        annotations = [{'segmentation': npz_info['mask'].tolist(),
                        'iscrowd': 0,
                        'image_id': image_id,
                        'bbox': bbox,
                        'category_id': id_from_name_map[category_name],
                        'category_name': category_name,
                        'occluder_box': occluder_box.tolist(),
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


image_info = main_dict['images']
# load image
for image in image_info:
    _image = skimage.io.imread(image['path'])
    plt.figure()
    plt.title("H x W={}x{}".format(image['height'], image['width']))
    plt.axis('off')
    plt.imshow(_image.astype(np.uint8))
    plt.show()

    annos = image['annotations'][0]
    mask = annos['segmentation']
    mask = np.array(mask)
    mask = (mask > 200)  # convert to boolean
    plt.figure()
    plt.title('segmentation:' + image['annotations'][0]['category_name'])
    plt.axis('off')
    plt.imshow(mask.astype(np.uint8))
    plt.show()

    occluder_mask = annos['occluder_mask']
    occluder_mask = np.array(occluder_mask)
    plt.figure()
    plt.title('occluder_mask:' + image['annotations'][0]['category_name'])
    plt.axis('off')
    plt.imshow(occluder_mask.astype(np.uint8))
    plt.show()

    box = annos['bbox']
    _image_temp = draw_box(_image, box, np.array([255, 0, 0]))
    plt.figure()
    plt.title("bbox")
    plt.axis('off')
    plt.imshow(_image_temp.astype(np.uint8))
    plt.show()

    _image_temp = _image.copy()
    for box in annos['occluder_box']:
        _image_temp = draw_box(_image, box, np.array([255, 0, 0]))
    plt.figure()
    plt.title("occluder_box")
    plt.axis('off')
    plt.imshow(_image_temp.astype(np.uint8))
    plt.show()

    box = annos['bbox']
    mini_mask = minimize_mask([box], np.expand_dims(mask, axis=-1), (56, 56))
    plt.figure()
    plt.title('mini_mask:' + image['annotations'][0]['category_name'])
    plt.axis('off')
    plt.imshow(mini_mask[:, :, 0].astype(np.uint8))
    plt.show()

    mask_expanded = expand_mask([box], mini_mask,
                                (image['height'], image['width']))
    plt.figure()
    plt.title('expanded_mask:' + image['annotations'][0]['category_name'])
    plt.axis('off')
    plt.imshow(mask_expanded[:, :, 0].astype(np.uint8))
    plt.show()
    break
