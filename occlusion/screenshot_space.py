import tensorflow as tf
import tensorlayer as tl
import numpy as np
from skimage.measure import find_contours
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import keras.backend as K
import keras.layers as KL

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.utils import expand_mask, resize_image
from mrcnn.utils_occlusion import mask2polygon


@tf.function
def extend_tensor(tensor, tensor_length, max_length):
    num_points_to_add = tf.cast(tf.math.ceil((max_length - tensor_length) / (tensor_length - 1)), tf.int32)
    extended_tensor = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    for i in tf.range(tensor_length - 1):
        p1 = tensor[i]
        p2 = tensor[i + 1]

        # CHANGED: Use tf.linspace to create an evenly distributed range of interpolation factors
        alpha = tf.linspace(0., 1., num_points_to_add + 2)
        # CHANGED: Generate new_points by interpolating between p1 and p2 using the alpha factors
        new_points = tf.cast(p1, tf.float32)[None, :] + alpha[:, None] * tf.cast(p2 - p1, tf.float32)[None, :]

        # CHANGED: Iterate over the new_points instead of the range of num_points_to_add + 1
        for j in tf.range(num_points_to_add + 1):
            extended_tensor = extended_tensor.write(extended_tensor.size(), tf.cast(new_points[j], tf.int32))

    # Add the last point
    extended_tensor = extended_tensor.write(extended_tensor.size(), tensor[-1])

    # Make sure the final tensor has exactly 100 points (in case of rounding issues)
    while tf.less(extended_tensor.size(), max_length):
        extended_tensor = extended_tensor.write(extended_tensor.size(), tensor[-1])

    return extended_tensor.stack()

