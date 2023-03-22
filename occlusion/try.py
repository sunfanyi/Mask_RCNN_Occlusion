# -*- coding: utf-8 -*-
# @File    : try.py
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
    if i == 1:
        break

mask_true = np.array(mask_true, dtype=np.bool)
a = np.argwhere(mask_true)
# b = tl.prepro.find_contours(mask_true)
# y = tf.placeholder(tf.float32, shape=[None])
polygon_true, _ = mask2polygon(mask_true, concat_verts=True)


@tf.function
def cap_and_average(tensor, max_length=100):
    tensor_length = tf.shape(tensor)[0]
    ratio = tf.cast(tensor_length, tf.float32) / max_length
    # return extend_tensor(tensor, tensor_length, max_length)
    case = tf.cond(ratio <= 1, lambda: 0, lambda: 1)
    return tf.switch_case(case, {0: lambda: extend_tensor(tensor, tensor_length, max_length),
                                 1: lambda: process_tensor(tensor, tensor_length, max_length, ratio)})

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

@tf.function
def extend_tensor(tensor, tensor_length, max_length):
    num_new_points = max_length - tensor_length
    # num_new_points = tf.cast(num_new_points, tf.int32)  # Cast to int32 instead of float32
    step = (tensor_length - 1) / (num_new_points + 1)
    step = tf.cast(step, tf.int32)

    if num_new_points > 0:
        positions = tf.cast(tf.round(tf.range(1, num_new_points + 1) * step), tf.int32)

        interp_points = (tf.gather(tensor, positions) + tf.gather(tensor, positions - 1)) / 2
        extended_tensor = tf.TensorArray(dtype=tensor.dtype, size=max_length)

        index = 0
        interp_idx = 0
        for i in range(max_length):
            if interp_idx < num_new_points and i == positions[interp_idx]:
                extended_tensor = extended_tensor.write(i, interp_points[interp_idx])
                interp_idx += 1
            else:
                extended_tensor = extended_tensor.write(i, tensor[index])
                index += 1

        return extended_tensor.stack()
    else:
        return tensor


mask = tf.placeholder(tf.float32, shape=[None, None, None])


def process_mask(mask_element):
    def case_zero():
        poly = tf.where(mask_element)
        poly = tf.cast(poly, tf.int32)
        poly_capped = cap_and_average(poly)
        poly_xy = tf.reverse(poly_capped, axis=[1])
        return poly_xy

    def case_one():
        return tf.zeros(shape=[0, 2], dtype=tf.int32)

    result = tf.cond(tf.greater(tf.shape(mask_element)[0], 0), case_zero, case_one)
    # result = tf.cond(tf.greater_equal(tf.shape(poly)[0], 40), case_zero, case_one)
    # result =
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