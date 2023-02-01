# -*- coding: utf-8 -*-
# @File    : occlusion.py
# @Time    : 30/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from pycocotools.coco import COCO

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class OcclusionConfig(Config):
    """Configuration for training on the occlusion dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "occlusion"

    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 13

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class OcclusionDataset(utils.Dataset):

    def load_occlusion(self, dataset_dir, subset, class_ids=None,
                       return_occlusion=False):
        occlusion = COCO(
            "{}/occlusion_coco_format_short.json".format(dataset_dir))
        # if subset == "minival" or subset == "valminusminival":
        #     subset = "val"
        image_dir = "{}/{}".format(dataset_dir, "images")

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(occlusion.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(occlusion.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(occlusion.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("occlusion", i, occlusion.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "occlusion", image_id=i,
                path=os.path.join(image_dir, occlusion.imgs[i]['file_name']),
                width=occlusion.imgs[i]["width"],
                height=occlusion.imgs[i]["height"],
                annotations=occlusion.loadAnns(occlusion.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_occlusion:
            return occlusion


dataset_dir = os.path.abspath('../../datasets/dataset_occluded')
dataset = OcclusionDataset()
occlusion = dataset.load_occlusion(dataset_dir, "train", return_occlusion=True)
dataset.prepare()