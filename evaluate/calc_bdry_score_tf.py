# -*- coding: utf-8 -*-
# @File    : calc_bdry_score_tf.py
# @Time    : 23/05/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

"""
Testing the boundary score calculation using tensorflow
"""

import tensorflow as tf
from skimage.measure import find_contours
import os
import sys
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.io

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.utils import expand_mask, resize_image, minimize_mask, extract_bboxes



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

    result = tf.reverse(result, axis=[1])  # yx to xy
    return result


@tf.function
def cap_tensor(tensor, max_length, ratio):
    """
    Cap the input tensor to max_length by searching two nearest values and averaging them
    """
    # Calculate the indices corresponding to the intervals
    indices = tf.range(max_length, dtype=tf.float32) * ratio

    lower = tf.cast(tf.math.floor(indices), tf.int32)  # Always round down
    upper = tf.cast(tf.math.ceil(indices), tf.int32)  # Always round up

    # Initialize an empty TensorArray to store the averaged values
    averaged_tensor = tf.TensorArray(dtype=tf.int32, size=max_length)

    for i in tf.range(max_length):
        # Extract values within the interval
        avg_value = tf.reduce_mean([tensor[lower[i], :], tensor[upper[i], :]], axis=0)

        averaged_tensor = averaged_tensor.write(i, avg_value)

    return averaged_tensor.stack()


@tf.function
def extend_tensor(tensor, tensor_length, max_length):
    """
    Extend the input tensor to max_length by linear interpolation.
    """
    num_points_to_add = tf.cast(tf.math.ceil((max_length - tensor_length) / (tensor_length - 1)), tf.int32)
    extended_tensor = tf.TensorArray(dtype=tf.int32, size=100, dynamic_size=True)

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
    if ~mask.any() or (mask.size == 0):
        return np.zeros((100, 2), dtype=np.float32)

    # Pad to ensure proper polygons for masks that touch image edges.
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    # contours = find_contours(padded_mask, 0.5)[0]
    contours = find_contours(padded_mask, 0.5)

    # Check if contours exist, if not, return an empty array
    if len(contours) == 0:
        return np.zeros((100, 2), dtype=np.float32)

    # flatten the contours
    # contours = np.concatenate(contours, axis=0).astype(np.float32)
    # choose the largest contour
    contours = max(contours, key=len).astype(np.float32)
    return contours


def mask2polygon(mask_element):
    def case_one():
        poly = tf.numpy_function(find_contours_wrapper, [mask_element], tf.float32)
        poly = tf.cast(poly, tf.int32)

        case = tf.equal(tf.shape(poly)[0], 100)
        poly_capped = tf.cond(case,
                              lambda: poly,
                              lambda: process_tensor_to_same_length(poly, 100))

        return poly_capped

    def case_two():
        return tf.zeros(shape=[100, 2], dtype=tf.int32)

    # mask_element = tf.cast(mask_element, tf.bool)
    result = tf.cond(tf.greater(tf.size(mask_element), 0), case_one, case_two)
    # result = tf.cond(tf.greater(tf.shape(mask_element)[0], 0), case_one, case_two)
    return result


# Calculate boundary score
def calc_bdry_score(args):
    polygon_true, polygon_pred, bbox_gt = args

    def _calc_score():
        # calculate the difference between polygons
        diff = polygon_pred[:, tf.newaxis, :] - polygon_true[tf.newaxis, :, :]
        diff = tf.cast(diff, tf.float32)

        # calculate all distances
        all_dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1))

        # find the minimum distance
        min_dist = tf.reduce_min(all_dist, axis=0)

        # average the minimum distances
        avg_dist = tf.reduce_mean(min_dist)

        rpn_area = tf.multiply(tf.subtract(bbox_gt[2], bbox_gt[0]),
                               tf.subtract(bbox_gt[3], bbox_gt[1]))
        rpn_area = tf.cast(rpn_area, tf.float32)
        bdry_score = tf.divide(avg_dist, rpn_area)

        # normalize the average minimum distance to get a score between 0 and 1
        bdry_score = tf.divide(1.0, tf.add(tf.multiply(6000.0, bdry_score), 1.0))
        return bdry_score

    # sometimes the polygons are empty after find_contours()
    def _check_all_zero(tensor):
        return tf.reduce_all(tf.equal(tensor, 0))
    both_empty = tf.logical_and(_check_all_zero(polygon_true),
                                _check_all_zero(polygon_pred))
    either_empty = tf.logical_or(_check_all_zero(polygon_true),
                                 _check_all_zero(polygon_pred))
    bdry_score = tf.cond(
        both_empty,
        lambda: 1.0,
        lambda: tf.cond(
            either_empty,
            lambda: 0.0,
            lambda: _calc_score()
        )
    )
    return bdry_score


def run_graph(mask_true, mask_pred, bbox_true):
    mask_true = np.array(mask_true, dtype=np.bool)
    mask_pred = np.array(mask_pred, dtype=np.bool)
    bbox_true = np.array(bbox_true, dtype=np.int32)

    mask_true_ph = tf.placeholder(tf.bool, shape=[None, None, None])
    mask_pred_ph = tf.placeholder(tf.bool, shape=[None, None, None])
    bbox_ph = tf.placeholder(tf.int32, shape=[None, 4])

    input_true = mask_true_ph
    input_pred = mask_pred_ph
    # a = tf.constant([], dtype=tf.bool)
    # input_pred = a

    def _calc_score(pred, true, bbox_true):
        polygons_pred = tf.map_fn(mask2polygon, pred, dtype=tf.int32)
        polygons_true = tf.map_fn(mask2polygon, true, dtype=tf.int32)
        bdry_score = tf.map_fn(calc_bdry_score,
                               (polygons_true, polygons_pred, bbox_true), dtype=tf.float32)
        return bdry_score, polygons_true, polygons_pred

    both_empty = tf.logical_and(tf.equal(tf.size(input_true), 0),
                                tf.equal(tf.size(input_pred), 0))
    either_empty = tf.logical_or(tf.equal(tf.size(input_true), 0),
                                 tf.equal(tf.size(input_pred), 0))
    bdry_score, polygons_true, polygons_pred = tf.cond(
        both_empty,
        lambda: (tf.constant(1.0), tf.zeros(shape=[100, 2], dtype=tf.int32),
                 tf.zeros(shape=[100, 2], dtype=tf.int32)),
        lambda: tf.cond(
            either_empty,
            lambda: (tf.constant(0.0), tf.zeros(shape=[100, 2], dtype=tf.int32),
                     tf.zeros(shape=[100, 2], dtype=tf.int32)),
            lambda: _calc_score(input_pred, input_true, bbox_true)
        )
    )

    # Run graph
    config = tf.ConfigProto()
    # config.log_device_placement = True
    config.allow_soft_placement = True  # allow CPU if GPU is not available
    with tf.Session(config=config) as sess:
        output_tensors = [bdry_score, polygons_true, polygons_pred]
        output_data = sess.run(output_tensors, feed_dict={mask_true_ph: mask_true,
                                                          mask_pred_ph: mask_pred,
                                                          bbox_ph: bbox_true
                                                          })

    bdry_score_output, polygons_true_output, polygons_pred_output = output_data

    return bdry_score_output, polygons_true_output, polygons_pred_output


if __name__ == '__main__':
    # prepare data
    from surgical_data import surgical

    dataset_dir = '../../datasets/3dStool'
    path = os.path.join(dataset_dir, 'train', 'manual_json', 'surgical_tool_train2020.json')
    image_dir = os.path.join(dataset_dir, 'train', 'surgical2020')
    raw_info = json.load(open(path))

    class_ids = None
    dataset_train = surgical.SurgicalDataset()
    sur = dataset_train.load_surgical(dataset_dir, "train", return_surgical=True,
                                      class_ids=class_ids)
    dataset_train.prepare()
    image_info = dataset_train.image_info

    N = len(dataset_train.image_info)

    ids = dataset_train.image_ids
    file_names = [i['path'] for i in image_info]
    mask_true = []
    bbox_true = []
    mask_pred = []
    bbox_pred = []

    for i in range(N):
        anno = image_info[i]['annotations'][0]
        mask, class_ids = dataset_train.load_mask(ids[i])

        # GT
        bboxes = extract_bboxes(mask)
        mask = minimize_mask(bboxes, mask, (56, 56))
        # gt_mask = np.transpose(gt_mask, (2, 0, 1))

        seg = mask.astype(np.float32)
        # seg = np.expand_dims(seg, -1)
        image_shape = (int(image_info[i]['height']), int(image_info[i]['width']))
        # image_shape = (1024, 1024)
        seg = expand_mask(bboxes, seg, image_shape)

        bbox_true.append(bboxes[0])
        mask_true.append(seg[:, :, 0])

        # GT
        gt_mask = seg[:, :, 0]
        if i == 0:
            pre = gt_mask.copy()

        bbox_pred.append(bboxes[0])
        if np.random.choice([True, False], p=[0.9, 0.1]):
            if np.random.choice([True, False], p=[0.9, 0.1]):
                # normal case
                mask_pred.append(gt_mask)
            else:
                print('wrong', i)
                # wrong value
                mask_pred.append(pre)
        else:
            print('empty', i)
            mask_pred.append(np.zeros_like(gt_mask, dtype=np.bool))

        pre = gt_mask.copy()

        if i == 200:
            break

    # run graph
    bdry_score_output, polygons_true_output, polygons_pred_output = \
        run_graph(mask_true, mask_pred, bbox_true)

    # # testing:
    # matplotlib.use('TkAgg')
    # for i in range(len(polygons_true_output)):
    #     # image = skimage.io.imread(os.path.join(image_dir, file_names[i]))
    #     image = skimage.io.imread(file_names[i])
    #     # image, _, _, _, _ = resize_image(image, min_dim=800, min_scale=0,
    #     #                                  max_dim=1024, mode='square')
    #
    #     # plot gt polygons
    #     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    #     # axes = get_ax(1, 2, size=6)
    #     axes[0].imshow(image)
    #     verts = polygons_true_output[i]
    #     xs, ys = zip(*verts)
    #     axes[0].scatter(xs, ys, c='r', s=1)
    #     axes[0].axis('off')
    #     axes[0].set_title('GT')
    #
    #     # plot predicted polygons
    #     axes[1].imshow(image)
    #     verts = polygons_pred_output[i]
    #     xs, ys = zip(*verts)
    #     axes[1].scatter(xs, ys, c='r', s=1)
    #     axes[1].axis('off')
    #     axes[1].set_title('Prediction')
    #     fig.suptitle('bdry_score = %.5f' % bdry_score_output[i])
    #     plt.show()





