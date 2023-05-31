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
import mrcnn.model as modellib

# Import Occlusion config
sys.path.append(os.path.join(ROOT_DIR, "surgical_data/"))
import surgical

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config = surgical.SurgicalConfig()
model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)

# Generate the plot
from tensorflow.keras.utils import plot_model

plot_model(model.keras_model, to_file='model_plot_surgical.png',
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
