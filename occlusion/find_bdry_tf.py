# -*- coding: utf-8 -*-
# @File    : find_bdry_tf.py
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


# coords = np.argwhere(mask > 0)
# coords = coords.reshape((-1, 2))  # N x 2
# coords = np.fliplr(coords)  # (y, x) to (x, y)
# tf.where
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
a = np.argwhere(mask_true)
# b = tl.prepro.find_contours(mask_true)
# y = tf.placeholder(tf.float32, shape=[None])
polygon_true, _ = mask2polygon(mask_true, concat_verts=True)


@tf.function
def cap_and_average(tensor, max_length=100):
    # return tensor
    tensor_length = tf.shape(tensor)[0]
    ratio = tf.cast(tensor_length, tf.float32) / max_length
    case = tf.greater(ratio, 1)
    result = tf.cond(case,
                     lambda: process_tensor(tensor, tensor_length, max_length, ratio),
                     lambda: extend_tensor(tensor, tensor_length, max_length))
    return result

@tf.function
def process_tensor(tensor, tensor_length, max_length, ratio):
    indices = tf.range(max_length, dtype=tf.float32) * ratio
    indices = tf.cast(tf.round(indices), tf.int32)

    starts = tf.concat([[0], indices[:-1]], axis=0)
    ends = indices

    averaged_tensor = tf.TensorArray(dtype=tensor.dtype, size=max_length)

    for i in tf.range(max_length):
        interval = tensor[starts[i]:ends[i], :]
        avg_value = tf.reduce_mean(interval, axis=0)
        averaged_tensor = averaged_tensor.write(i, avg_value)

    return averaged_tensor.stack()


# this has no error but extended points in same locations
# @tf.function
# def extend_tensor(tensor, tensor_length, max_length):
#     num_new_points = max_length - tensor_length
#     if tf.greater(num_new_points, 0):
#         step = (tensor_length - 1) / (num_new_points + 1)
#         step = tf.cast(step, tf.int32)
#
#         positions = tf.cast(tf.round(tf.range(1, num_new_points + 1) * step), tf.int32)
#         positions = tf.clip_by_value(positions, 0, tensor_length - 2)
#
#         interp_points = (tf.gather(tensor, positions) + tf.gather(tensor, positions - 1)) / 2
#         interp_points = tf.cast(interp_points, tf.int32)
#
#         extended_tensor = tf.concat([tensor[:positions[0]], interp_points[0:1]], axis=0)
#         for i in range(1, num_new_points):
#             extended_tensor = tf.concat([extended_tensor, tensor[positions[i - 1]:positions[i]], interp_points[i:i + 1]], axis=0)
#         extended_tensor = tf.concat([extended_tensor, tensor[positions[-1]:]], axis=0)
#
#         return extended_tensor
#     else:
#         return tensor



# @tf.function
# def extend_tensor(tensor, tensor_length, max_length):
#     num_points_to_add = tf.cast(tf.math.ceil((max_length - tensor_length) / (tensor_length - 1)), tf.int32)
#     extended_tensor = tf.TensorArray(dtype=tf.int32, size=max_length)
#
#     counter = tf.constant(0, dtype=tf.int32)
#
#     for i in tf.range(tensor_length - 1):
#         p1 = tensor[i]
#         p2 = tensor[i + 1]
#
#         diff = p2 - p1
#
#         for j in tf.range(num_points_to_add + 1):
#             alpha = tf.cast(j, tf.float32) / tf.cast(num_points_to_add + 1, tf.float32)
#             new_point = tf.cast(p1, tf.float32) + alpha * tf.cast(diff, tf.float32)
#
#             if tf.less(counter, max_length):
#                 extended_tensor = extended_tensor.write(counter, tf.cast(new_point, tf.int32))
#                 counter += 1
#
#     # Add the last point
#     while tf.less(counter, max_length):
#         extended_tensor = extended_tensor.write(counter, tensor[-1])
#         counter += 1
#
#     # # Add the last point
#     # if tf.less(counter, max_length):
#     #     extended_tensor = extended_tensor.write(counter, tensor[-1])
#
#     return extended_tensor.stack()


@tf.function
def extend_tensor(tensor, tensor_length, max_length):
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

    # Make sure the final tensor has exactly 100 points (in case of rounding issues)
    while tf.less(extended_tensor.size(), max_length):
        extended_tensor = extended_tensor.write(extended_tensor.size(), tensor[-1])

    # Changes: Randomly remove points to make the final tensor have exactly 100 points
    def true_fn():
        indices_to_keep = tf.random.shuffle(tf.range(extended_tensor.size()))[:max_length]
        indices_to_keep = tf.sort(indices_to_keep)
        return tf.gather(extended_tensor.stack(), indices_to_keep, axis=0)

    def false_fn():
        return extended_tensor.stack()

    final_extended_tensor = tf.cond(tf.greater(extended_tensor.size(), max_length), true_fn, false_fn)


    return final_extended_tensor



mask = tf.placeholder(tf.float32, shape=[None, None, None])


def find_contours_wrapper(mask_element):
    contours = find_contours(mask_element, 0.5)[0]
    return np.array(contours, dtype=np.float32)  # output as numpy array
    # return np.array(contours, dtype=np.float32)[::50]  # for testing


def process_mask(mask_element):
    def case_zero():
        # poly = tf.where(mask_element)
        # poly = tf.cast(poly, tf.int32)
        # poly = tl.prepro.find_contours(mask_element)
        poly = tf.py_func(find_contours_wrapper, [mask_element], tf.float32)
        poly = tf.cast(poly, tf.int32)
        poly_capped = cap_and_average(poly)
        poly_xy = tf.reverse(poly_capped, axis=[1])
        return poly_xy

    def case_one():
        return tf.zeros(shape=[0, 2], dtype=tf.int32)

    result = tf.cond(tf.greater(tf.shape(mask_element)[0], 0), case_zero, case_one)
    return result


y = tf.map_fn(process_mask, mask, dtype=tf.int32)


with tf.Session() as sess:
    input_data = mask_true
    output_data = sess.run(y, feed_dict={mask: input_data})
    print(output_data)



# xy to flat
contours_flat = output_data.reshape(output_data.shape[0], -1)
# contours_flat = []  # 2n x 1
# for i in range(output_data.shape[0]):  # for each segmentation
#     contour = output_data[i]
#     contours_flat.append([item for sublist in contour for item in sublist])


# plot to check if any loss in accuracy
for i in range(len(output_data)):
    image = skimage.io.imread(os.path.join(image_dir, file_names[i]))
    image, _, _, _, _ = resize_image(image, min_dim=800, min_scale=0,
                                     max_dim=1024, mode='square')

    # plot gt polygons
    fig, ax = plt.subplots()
    # axes = get_ax(1, 2, size=6)
    ax.imshow(image)
    verts = output_data[i]
    # verts = verts[::10]  # make points less dense
    xs, ys = zip(*verts)
    ax.scatter(xs, ys, c='r', s=1)
    ax.axis('off')
    ax.set_title('GT')
    fig.show()