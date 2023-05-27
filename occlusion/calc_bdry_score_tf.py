# -*- coding: utf-8 -*-
# @File    : calc_bdry_score_tf.py
# @Time    : 06/03/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

"""
Testing the boundary score calculation using tensorflow (depreciated)
"""

import tensorflow as tf
from skimage.measure import find_contours
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.utils import expand_mask, resize_image


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


def find_contours_wrapper(mask):
    # if empty mask, return empty array
    if ~mask.any():
        return np.array([[0, 0]], dtype=np.float32)

    # Pad to ensure proper polygons for masks that touch image edges.
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)[0]
    return np.array(contours, dtype=np.float32)  # output as numpy array
    # return np.array(contours, dtype=np.float32)[::50]  # for testing


def mask2polygon(mask_element):
    def case_one():
        poly = tf.numpy_function(find_contours_wrapper, [mask_element], tf.float32)
        poly = tf.cast(poly, tf.int32)
        poly_capped = process_tensor_to_same_length(poly)
        poly_xy = tf.reverse(poly_capped, axis=[1])  # yx to xy
        return poly_xy

    def case_two():
        return tf.zeros(shape=[0, 2], dtype=tf.int32)

    # mask_element = tf.cast(mask_element, tf.bool)
    result = tf.cond(tf.greater(tf.shape(mask_element)[0], 0), case_one, case_two)
    return result


# Calculate boundary score
def calc_bdry_score(args):
    polygon_true, polygon_pred = args

    # search for the cloest point
    diff = polygon_pred[:, tf.newaxis, :] - polygon_true[tf.newaxis, :, :]
    diff = tf.cast(diff, tf.float32)
    all_dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1))
    min_dist = tf.reduce_min(all_dist, axis=1)  # [num_points]
    bdry_score = tf.reduce_mean(min_dist) / tf.cast(tf.shape(min_dist)[0], tf.float32)
    return bdry_score


def run_graph(mask_true, mask_pred):
    mask_true = np.array(mask_true, dtype=np.bool)
    mask_pred = np.array(mask_pred, dtype=np.bool)

    mask_true_ph = tf.placeholder(tf.bool, shape=[None, None, None])
    mask_pred_ph = tf.placeholder(tf.bool, shape=[None, None, None])

    polygons_true = tf.map_fn(mask2polygon, mask_true_ph, dtype=tf.int32)
    polygons_pred = tf.map_fn(mask2polygon, mask_pred_ph, dtype=tf.int32)
    bdry_score = tf.map_fn(calc_bdry_score, (polygons_true, polygons_pred), dtype=tf.float32)

    # Run graph
    with tf.Session() as sess:
        output_tensors = [bdry_score, polygons_true, polygons_pred]
        output_data = sess.run(output_tensors, feed_dict={mask_true_ph: mask_true,
                                                          mask_pred_ph: mask_pred})

        bdry_score_output, polygons_true_output, polygons_pred_output = output_data
    return bdry_score_output, polygons_true_output, polygons_pred_output


if __name__ == '__main__':
    # prepare data
    image_dir = '../../datasets/dataset_occluded/images'

    sample_info = json.load(open('sample_mask_info.json', 'r'))
    N = len(sample_info)

    ids = [i['id'] for i in sample_info]
    file_names = [i['file_names'] for i in sample_info]
    mask_true = []
    bbox_true = []
    mask_pred = []
    bbox_pred = []

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

        # predicted
        bbox = sample_info[i]['bbox_pred']
        bbox = np.expand_dims(bbox, 0).astype('int')

        seg = sample_info[i]['mask_pred']
        seg = np.expand_dims(seg, -1)
        image_shape = (int(sample_info[i]['height']), int(sample_info[i]['width']))
        seg = expand_mask(bbox, seg, image_shape)

        bbox_pred.append(bbox[0])
        mask_pred.append(seg[:, :, 0])

        if i == 3:
            break

    # run graph
    bdry_score_output, polygons_true_output, polygons_pred_output = \
        run_graph(mask_true, mask_pred)

    # testing:
    for i in range(len(polygons_true_output)):
        image = skimage.io.imread(os.path.join(image_dir, file_names[i]))
        image, _, _, _, _ = resize_image(image, min_dim=800, min_scale=0,
                                         max_dim=1024, mode='square')

        # plot gt polygons
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # axes = get_ax(1, 2, size=6)
        axes[0].imshow(image)
        verts = polygons_true_output[i]
        xs, ys = zip(*verts)
        axes[0].scatter(xs, ys, c='r', s=1)
        axes[0].axis('off')
        axes[0].set_title('GT')

        # plot predicted polygons
        axes[1].imshow(image)
        verts = polygons_pred_output[i]
        xs, ys = zip(*verts)
        axes[1].scatter(xs, ys, c='r', s=1)
        axes[1].axis('off')
        axes[1].set_title('Prediction')
        fig.suptitle('bdry_score = %.5f' % bdry_score_output[i])
        plt.show()
