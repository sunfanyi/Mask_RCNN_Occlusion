# -*- coding: utf-8 -*-
# @File    : plot_model.py
# @Time    : 09/01/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import os
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import Occlusion config
sys.path.append(os.path.join(ROOT_DIR, "occlusion/"))
from occlusion import occlusion
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco
# Import shape config
sys.path.append(os.path.join(ROOT_DIR, "samples/shapes/"))
import shapes

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# # Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# # Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# class InferenceConfig(coco.CocoConfig):
#     # Set batch size to 1 since we'll be running inference on
#     # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1

config = occlusion.OcclusionConfig()
# config = coco.CocoConfig()
# config = shapes.ShapesConfig()


# config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
# model.load_weights(COCO_MODEL_PATH, by_name=True)


# Generate the plot
from tensorflow.keras.utils import plot_model

plot_model(model.keras_model, to_file='model_plot_occlusion.png',
           show_shapes=True,
           show_layer_names=True)


# change to keras model rather than maskrcnn model because maskrcnn is a custom class
# plot_model(model.keras_model, to_file='model_plot.png',show_shapes=True)

# # Show the plot here
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# img = mpimg.imread('model_plot.png')
# plt.figure(figsize=(30,15))
# imgplot = plt.imshow(img)
