# -*- coding: utf-8 -*-
# @File    : occluded.py
# @Time    : 30/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import os
import numpy as np

dataset_dir = 'D:\Desktop\FYP - Robust Surgical Tool Detection and Occlusion Handling using Deep Learning\datasets\dataset_occluded'

example_annos = np.load(os.path.join(dataset_dir, 'annotations/bottleFGL1_BGL1/n02823428_56.npz'),
                allow_pickle=True)
print(example_annos.files)
print(example_annos['source'])

annotation_file_dir = os.path.join(dataset_dir, 'annotations')
# print(os.walk(annotation_file_dir))
lst = os.listdir(annotation_file_dir)
files = []
for dirpath, dirnames, filenames in os.walk(annotation_file_dir):
    for filename in filenames:
        files.append(os.path.join(dirpath, filename))
        # get image id from the file name

class_info = [{"source": "", "id": 0, "name": "BG"}]
image_info_dir = []

class_ids = []
class_names = []
image_ids = []
for file in files[:100]:
    image_ids.append(filename.split('.')[0].split('n')[1])
    annos = np.load(file, allow_pickle=True)
    # if str(annos['category']) not in class_map.values():
    if str(annos['category']) not in class_names:
        class_names.append(str(annos['category']))


# def add_image(source, image_id, path, image_info_dir, **kwargs):
#     image_info = {
#         "id": image_id,
#         "source": source,
#         "path": path,
#     }
#     image_info.update(kwargs)
#     image_info_dir.append(image_info)
#     return image_info_dir
#
#
# for i in image_ids:
#     image_info_dir = add_image(
#         "occlusion", image_id=i,
#         path=os.path.join(image_dir, coco.imgs[i]['file_name']),
#         width=coco.imgs[i]["width"],
#         height=coco.imgs[i]["height"],
#         annotations=coco.loadAnns(coco.getAnnIds(
#             imgIds=[i], catIds=class_ids, iscrowd=None)))

class_ids = np.arange(len(class_names) + 1)  # 0th is the background
class_names = ['BG'] + class_names

for i in range(1, len(class_names)):
    class_info.append({
        "source": 'occlusion',
        "id": class_ids[i],
        "name": class_names[i],
    })
