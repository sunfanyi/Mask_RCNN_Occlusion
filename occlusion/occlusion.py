# -*- coding: utf-8 -*-
# @File    : occlusion.py
# @Time    : 30/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('tkagg')

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize

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
                       return_occlusion=False, mask_format='polygon'):
        if mask_format == 'polygon':
            occlusion = COCO(
                "{}/occlusion_final.json".format(dataset_dir))
        elif mask_format == 'bitmap':
            occlusion = COCO(
                "{}/occlusion_bitmap_short.json".format(dataset_dir))
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


    def load_mask(self, image_id, mask_format='polygon'):
        image_info = self.image_info[image_id]
        if image_info["source"] != "occlusion":
            return super(OcclusionDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "occlusion.{}".format(annotation['category_id']))
            if class_id:
                if mask_format == 'polygon':
                    m = self.annToMask(annotation, image_info["height"],
                                       image_info["width"])
                elif mask_format == 'bitmap':
                    m = np.array(annotation['segmentation'])
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
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(OcclusionDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "occlusion":
            return info['id']
        else:
            super(OcclusionDataset, self).image_reference(image_id)


    def annToRLE(self, ann, height, width):
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


    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


if __name__ == '__main__':
    dataset_dir = os.path.abspath('../../datasets/dataset_occluded')
    dataset = OcclusionDataset()
    occlusion = dataset.load_occlusion(dataset_dir, "train", return_occlusion=True)
    dataset.prepare()


