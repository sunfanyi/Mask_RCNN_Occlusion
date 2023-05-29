# -*- coding: utf-8 -*-
# @File    : surgical.py
# @Time    : 22/05/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import os
import sys
import numpy as np
import imgaug

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# ROOT_DIR = os.path.abspath("../")
ROOT_DIR = "/rds/general/user/fs1519/home/FYP/Mask_RCNN-Occlusion"
# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class SurgicalConfig(Config):
    """Configuration for training on the surgical dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "surgical"

    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class SurgicalDataset(utils.Dataset):

    def load_surgical(self, dataset_dir, subset, class_ids=None,
                       return_surgical=False):
        surgical = COCO(
            "{}/{}/manual_json/surgical_tool_{}2020.json".format(dataset_dir, subset, subset))

        image_dir = "{}/{}/surgical2020".format(dataset_dir, subset)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(surgical.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(surgical.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(surgical.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("surgical", i, surgical.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "surgical", image_id=i,
                path=os.path.join(image_dir, surgical.imgs[i]['file_name']),
                width=surgical.imgs[i]["width"],
                height=surgical.imgs[i]["height"],
                annotations=surgical.loadAnns(surgical.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_surgical:
            return surgical


    def load_mask(self, image_id, mask_format='polygon'):
        image_info = self.image_info[image_id]
        if image_info["source"] != "surgical":
            return super(SurgicalDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "surgical.{}".format(annotation['category_id']))
            if class_id:
                if mask_format == 'polygon':
                    m = self.annToMask(annotation, image_info["height"], image_info["width"])

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
            return super(SurgicalDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "surgical":
            return info['id']
        else:
            super(SurgicalDataset, self).image_reference(image_id)

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


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on surgical dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the Surgical dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--occluded_only', required=False,
                        default=False,
                        metavar="<train all images or occluded only>",
                        help='Images used for training, all or occluded only')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--info', required=False,
                        default=None,
                        metavar="training information",
                        help='training information')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Info: ", args.info)

    # Configurations
    if args.command == "train":
        config = SurgicalConfig()
    else:
        class InferenceConfig(SurgicalConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()

    if args.occluded_only:
        class_ids = list(range(1, 13))
        config.NUM_CLASSES -= 1
        config.__init__()
    else:
        class_ids = None

    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    if args.model.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as in the Mask RCNN paper.
        dataset_train = SurgicalDataset()
        dataset_train.load_surgical(args.dataset, "train", class_ids=class_ids)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = SurgicalDataset()
        val_type = "val"
        dataset_val.load_surgical(args.dataset, val_type, class_ids=class_ids)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # # Training - Stage 1
        # print("Training Stage 1")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=70,
        #             layers='all',
        #             augmentation=augmentation,
        #             info=args.info)
        #
        # # Training - Stage 2
        # print("Training Stage 2")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE / 10,
        #             epochs=100,
        #             layers='all',
        #             augmentation=augmentation,
        #             info=args.info)

        # # Training - Stage 1
        # print("Training Stage 1")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=60,
        #             layers='heads',
        #             augmentation=augmentation,
        #             info=args.info)
        #
        # # Training - Stage 2
        # print("Training Stage 2")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=80,
        #             layers='4+',
        #             augmentation=augmentation,
        #             info=args.info)
        #
        # # Training - Stage 3
        # # Fine tune all layers
        # print("Training Stage 3")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE / 10,
        #             epochs=120,
        #             layers='all',
        #             augmentation=augmentation,
        #             info=args.info)

        # Training - Stage 1
        # print("Training Stage 1")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=15,
        #             layers='all',
        #             augmentation=augmentation,
        #             info=args.info)
        #
        # # Training - Stage 2
        # print("Training Stage 2")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE / 10,
        #             epochs=21,
        #             layers='all',
        #             augmentation=augmentation,
        #             info=args.info)
        #
        # # Training - Stage 3
        # # Fine tune all layers
        # print("Training Stage 3")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE / 100,
        #             epochs=24,
        #             layers='all',
        #             augmentation=augmentation,
        #             info=args.info)

        # Training - Stage 1
        print("Training Stage 1")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=75,
                    layers='all',
                    augmentation=augmentation,
                    info=args.info)

        # Training - Stage 2
        print("Training Stage 2")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=105,
                    layers='all',
                    augmentation=augmentation,
                    info=args.info)

        # Training - Stage 3
        # Fine tune all layers
        print("Training Stage 3")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 100,
                    epochs=120,
                    layers='all',
                    augmentation=augmentation,
                    info=args.info)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = SurgicalDataset()
        val_type = "val" if args.year in '2017' else "minival"
        coco = dataset_val.load_surgical(args.dataset, val_type)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        # evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))

