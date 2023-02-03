# -*- coding: utf-8 -*-
# @File    : make_json_occlusion_coco.py
# @Time    : 01/02/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

# The raw annotation files are very slow to read (in npz). Therefore json files
# will be created and useful information will be written into it.
# This file saves all the data in coco format

import os
import json
import numpy as np
import skimage.io

from pycocotools import mask
from pycococreatortools.pycococreatortools import binary_mask_to_polygon, \
    resize_binary_mask

dataset_dir = '../../datasets/dataset_occluded'
# dataset_dir = '../check_mask'

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


def create_annotation_info(annotation_id, image_id, category_id, binary_mask,
                           image_size=None, tolerance=2, bbox=None,
                           polygon_mask=True):
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
    segmentation = binary_mask_to_polygon(binary_mask, tolerance)
    if not segmentation:
        return None

    m = segmentation if polygon_mask else binary_mask.tolist()

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


def add_image_to_list(main_dict, file_name, par_dir, anno_id,
                      polygon_mask=True):
    annos_dir = os.path.join(dataset_dir, 'annotations')
    images_dir = os.path.join(dataset_dir, 'images')

    image_id = file_name.split('.')[0]
    image_path = os.path.join(images_dir, par_dir, file_name)

    image_id = par_dir + '/' + image_id
    file_name = par_dir + '/' + file_name

    # get height and width from jpeg
    image = skimage.io.imread(image_path)
    image_size = image.shape[:2]

    image_info = create_image_info(image_id, file_name, image_size)
    main_dict['images'].append(image_info)

    # get annotation information from npz
    npz_path = os.path.join(annos_dir, image_id) + '.npz'
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
                                             mask, image_size=image_size,
                                             tolerance=2, bbox=bbox,
                                             polygon_mask=polygon_mask)
    main_dict['annotations'].append(annotation_info)
    anno_id += 1
    # for occluder_mask in occluder_masks:
    annotation_info = create_annotation_info(anno_id, image_id,
                                             id_from_name_map['occluder'],
                                             occluder_mask,
                                             image_size=image_size, tolerance=2,
                                             bbox=occluder_box,
                                             polygon_mask=polygon_mask)
    main_dict['annotations'].append(annotation_info)
    anno_id += 1
    return main_dict, anno_id


def write_to_json(train_val_split=False, debug=False, write=True):
    """
    train_val_split:
        folders names represent occlusion level, for example
        aeroplaneFGL1_BGL1 has: images 1, 2, 3, 4, 5
        aeroplaneFGL1_BGL2 has: images 1, 2, 3, 5, 7
        aeroplaneFGL1_BGL3 has: images 1, 2, 3, 4, 6

        Images with same id in different folders have same occluded mask but
        different occlusion level, and therefore have to be in either training
        set or val set together.

        aeroplaneFGL1_BGL1 is used as a reference for selecting validation set.
        for example, images labelled with 1 -4 for training and 5 for validation
    """
    lists_dir = os.path.join(dataset_dir, 'lists')
    if train_val_split:
        if debug:
            lists_dir = os.path.join(dataset_dir, 'lists_debug_train_val_split')
        train_dict = {'images': [], 'annotations': [],
                      'categories': cateogories}
        val_dict = {'images': [], 'annotations': [], 'categories': cateogories}

        folders_per_cat = 3 if debug else 9
        # set initial value as 9 to read from the first category
        num_folders = folders_per_cat

        train_val_ratio = 4  # train:val ~= 20:1
        anno_id = 1
        for lst in os.listdir(lists_dir):
            # for each par_dir
            par_dir = lst.split('.')[0]
            with open(os.path.join(lists_dir, lst)) as file:
                file_names = file.readlines()
            file_names = [i.strip() for i in file_names]
            if num_folders // folders_per_cat:  # finish browsing one category
                # traverse to the next category and update val_id_list
                num_folders = 0
                val_id_list = file_names[:len(file_names) // train_val_ratio]
            else:
                num_folders += 1

            for file_name in file_names:
                # for each image
                file_name = file_name.strip()

                if file_name in val_id_list:
                    val_dict, anno_id = add_image_to_list(val_dict, file_name,
                                                          par_dir, anno_id,
                                                          polygon_mask=True)
                else:
                    train_dict, anno_id = add_image_to_list(train_dict,
                                                            file_name,
                                                            par_dir, anno_id,
                                                            polygon_mask=True)

        if debug:
            target_path_train = os.path.join(dataset_dir,
                                             "occlusion_train_short.json")
            target_path_val = os.path.join(dataset_dir,
                                           "occlusion_val_short.json")
        else:
            target_path_train = os.path.join(dataset_dir,
                                             "occlusion_train.json")
            target_path_val = os.path.join(dataset_dir,
                                           "occlusion_val.json")

        if write:
            with open(target_path_train, "w") as outfile:
                json.dump(train_dict, outfile)
            with open(target_path_val, "w") as outfile:
                json.dump(val_dict, outfile)

        return train_dict, val_dict

    else:
        main_dict = {'images': [], 'annotations': [], 'categories': cateogories}
        anno_id = 1
        for lst in os.listdir(lists_dir):
            # for each par_dir
            par_dir = lst.split('.')[0]
            with open(os.path.join(lists_dir, lst)) as file:
                file_names = file.readlines()
            file_names = [i.strip() for i in file_names]
            for file_name in file_names:
                # for each image
                main_dict, anno_id = add_image_to_list(main_dict, file_name,
                                                       par_dir, anno_id,
                                                       polygon_mask=True)
                if debug:
                    # save one image from each folder
                    break
            # if debug:
            #     # save all images from the first folder
            #     break

        if debug:
            target_path = os.path.join(dataset_dir,
                                       "occlusion_short.json")
        else:
            target_path = os.path.join(dataset_dir,
                                       "occlusion_full.json")

        if write:
            with open(target_path, "w") as outfile:
                json.dump(main_dict, outfile)

        return main_dict


if __name__ == "__main__":
    # train_val_split = True
    res = write_to_json(train_val_split=True, debug=False, write=True)
