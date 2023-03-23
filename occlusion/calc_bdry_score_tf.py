# -*- coding: utf-8 -*-
# @File    : calc_bdry_score_tf.py
# @Time    : 06/03/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

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

image_dir = '../../datasets/dataset_occluded/images'


sample_info = json.load(open('sample_mask_info.json', 'r'))
N = len(sample_info)

ids = [i['id'] for i in sample_info]
file_names = [i['file_names'] for i in sample_info]
mask_true = []
bbox_true = []

for i in range(N):
    # GT
    bbox = sample_info[i]['bbox_true']
    bbox = np.expand_dims(bbox, 0).astype('int')

    seg = sample_info[i]['mask_true']
    seg = np.expand_dims(seg, -1)
    image_shape = (int(sample_info[i]['height']), int(sample_info[i]['width']))
    seg = expand_mask(bbox, seg, image_shape)

    bbox_true.append(bbox[0])
    mask_true.append(seg[:, :, 0])
    if i == 3:
        break

mask_true = np.array(mask_true, dtype=np.bool)
polygon_true, _ = mask2polygon(mask_true, concat_verts=True)


@tf.function
def process_tensor_to_same_length(tensor, max_length=100):
    """
    Define a function that processes input tensor to have the same length as the given max_length
    If the input tensor's length is greater than max_length, it will be capped using cap_tensor function
    If the input tensor's length is less than max_length, it will be extended using extend_tensor function
    """
    tensor_length = tf.shape(tensor)[0]
    ratio = tf.cast(tensor_length, tf.float32) / max_length
    case = tf.greater(ratio, 1)
    result = tf.cond(case,
                     lambda: cap_tensor(tensor, max_length, ratio),
                     lambda: extend_tensor(tensor, tensor_length, max_length))
    return result


@tf.function
def cap_tensor(tensor, max_length, ratio):
    """
    Cap the input tensor to max_length by averaging the values in each interval
    """
    # Calculate the indices corresponding to the intervals
    indices = tf.range(max_length, dtype=tf.float32) * ratio
    indices = tf.cast(tf.round(indices), tf.int32)

    indices, _ = tf.unique(indices)  # remove duplicates

    starts = tf.concat([[0], indices[:-1]], axis=0)
    ends = indices

    # Initialize an empty TensorArray to store the averaged values
    averaged_tensor = tf.TensorArray(dtype=tensor.dtype, size=max_length)

    for i in tf.range(max_length):
        # Extract values within the interval
        interval = tensor[starts[i]:ends[i], :]
        # Compute average
        avg_value = tf.reduce_mean(interval, axis=0)
        averaged_tensor = averaged_tensor.write(i, avg_value)

    return averaged_tensor.stack()


@tf.function
def extend_tensor(tensor, tensor_length, max_length):
    """
    Extend the input tensor to max_length by linear interpolation.
    """
    num_points_to_add = tf.cast(tf.math.ceil((max_length - tensor_length) / (tensor_length - 1)), tf.int32)
    extended_tensor = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    for i in tf.range(tensor_length - 1):
        p1 = tensor[i]
        p2 = tensor[i + 1]

        alpha = tf.linspace(0., 1., num_points_to_add + 2)
        new_points = tf.cast(p1, tf.float32)[None, :] + alpha[:, None] * tf.cast(p2 - p1, tf.float32)[None, :]

        for j in tf.range(num_points_to_add + 1):
            extended_tensor = extended_tensor.write(extended_tensor.size(), tf.cast(new_points[j], tf.int32))

    # Add the last point
    extended_tensor = extended_tensor.write(extended_tensor.size(), tensor[-1])

    # Make sure the final tensor has no less than 100 points (in case of rounding issues)
    while tf.less(extended_tensor.size(), max_length):
        extended_tensor = extended_tensor.write(extended_tensor.size(), tensor[-1])

    # Make sure it has no more than 100 points:
    def true_fn():
        # Randomly remove points to make the final tensor have exactly 100 points
        indices_to_keep = tf.random.shuffle(tf.range(extended_tensor.size()))[:max_length]
        indices_to_keep = tf.sort(indices_to_keep)
        return tf.gather(extended_tensor.stack(), indices_to_keep, axis=0)

    def false_fn():
        return extended_tensor.stack()

    final_extended_tensor = tf.cond(tf.greater(extended_tensor.size(), max_length), true_fn, false_fn)

    return final_extended_tensor


mask = tf.placeholder(tf.bool, shape=[None, None, None])


def find_contours_wrapper(mask_element):
    contours = find_contours(mask_element, 0.5)[0]
    return np.array(contours, dtype=np.float32)  # output as numpy array
    # return np.array(contours, dtype=np.float32)[::50]  # for testing


def process_mask(mask_element):
    def case_zero():
        poly = tf.py_func(find_contours_wrapper, [mask_element], tf.float32)
        poly = tf.cast(poly, tf.int32)
        poly_capped = process_tensor_to_same_length(poly)
        poly_xy = tf.reverse(poly_capped, axis=[1])  # yx to xy
        return poly_xy

    def case_one():
        return tf.zeros(shape=[0, 2], dtype=tf.int32)

    # mask_element = tf.cast(mask_element, tf.bool)
    result = tf.cond(tf.greater(tf.shape(mask_element)[0], 0), case_zero, case_one)
    return result


y = tf.map_fn(process_mask, mask, dtype=tf.int32)


with tf.Session() as sess:
    input_data = mask_true
    output_data = sess.run(y, feed_dict={mask: input_data})
    print(output_data)


# testing:
for i in range(len(output_data)):
    image = skimage.io.imread(os.path.join(image_dir, file_names[i]))
    image, _, _, _, _ = resize_image(image, min_dim=800, min_scale=0,
                                     max_dim=1024, mode='square')

    # plot gt polygons
    fig, ax = plt.subplots()
    ax.imshow(image)
    verts = output_data[i]
    xs, ys = zip(*verts)
    ax.scatter(xs, ys, c='r', s=1)
    ax.axis('off')
    ax.set_title('GT')
    fig.show()
