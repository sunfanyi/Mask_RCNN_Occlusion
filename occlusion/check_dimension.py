# -*- coding: utf-8 -*-
# @File    : check_dimension.py
# @Time    : 25/02/2023
# @Author  : Fanyi Sun
# @Github  : https://github.com/sunfanyi
# @Software: PyCharm

import os
import sys
import numpy as np

import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from occlusion import *

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model as modellib, utils

ROOT_DIR = os.path.abspath("../")
# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logstemp")


config = OcclusionConfig()
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=DEFAULT_LOGS_DIR)

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

smooth = 1e-7

print('\n===============================================================\n')
target_masks = KL.Input(batch_shape=[5, 200, 28, 28], name="target_masks")
target_class_ids = KL.Input(batch_shape=[5, 200], name="target_class_ids")
pred_masks = KL.Input(batch_shape=[5, 200, 28, 28, 6], name="pred_masks")
pred_bdry_score = KL.Input(batch_shape=[5, 200, 1, 1, 6], name="pred_bdry_score")
print(target_masks, '-> gt_mask from DetectionTarget Layer')
print(target_class_ids, '-> gt_class_id from DetectionTarget Layer')
print(pred_masks, '-> predicted_mask')
print(pred_bdry_score, '-> predicted_bdry_score')

print('\n===============================================================\n')
# Reshape for simplicity. Merge first two dimensions into one.
target_class_ids = K.reshape(target_class_ids, (-1,))  # [N]
print(target_class_ids, '-> target_class_ids')
mask_shape = tf.shape(target_masks)
target_masks = K.reshape(target_masks,
                         (-1, mask_shape[2],
                          mask_shape[3]))  # [N, height, width]
print(target_masks, '-> target_masks')
pred_shape = tf.shape(pred_masks)
pred_masks = K.reshape(pred_masks,
                       (-1, pred_shape[2], pred_shape[3],
                        pred_shape[4]))  # [N, height, width, num_classes]
print(pred_masks, '-> pred_masks')
pred_bdry_score = tf.squeeze(pred_bdry_score, axis=[2, 3])
print(pred_bdry_score, '-> pred_bdry_score')
pred_bdry_shape = tf.shape(pred_bdry_score)
pred_bdry_score = K.reshape(pred_bdry_score,
                            (-1, pred_bdry_shape[2]))  # [N, num_classes]
print(pred_bdry_score, '-> pred_bdry_score')
# Permute predicted masks to [N, num_classes, height, width]
pred_masks = tf.transpose(pred_masks,
                          [0, 3, 1, 2])  # [N, num_classes, height, width]
print(pred_masks, '-> pred_masks')


print('\n===============================================================\n')
# Only positive ROIs contribute to the loss. And only
# the class specific mask of each ROI.
positive_ix = tf.where(target_class_ids > 0)[:, 0]
print(positive_ix, '-> positive_ix')
positive_class_ids = tf.cast(
    tf.gather(target_class_ids, positive_ix), tf.int64)
print(positive_class_ids, '-> positive_class_ids')
indices = tf.stack([positive_ix, positive_class_ids], axis=1)
print(indices, '-> indices')


print('\n===============================================================\n')
# Gather the masks (predicted and true) that contribute to loss
mask_true = tf.gather(target_masks, positive_ix)  # [N, h, w]
print(mask_true, '-> mask_true')
print('Fix the shape:')
mask_true.set_shape((150, 28, 28))
print(mask_true, '-> mask_true')

mask_pred = tf.gather_nd(pred_masks, indices)  # [N, h, w]
print(mask_pred, '-> mask_pred')
print('Fix the shape:')
mask_pred.set_shape((150, 28, 28))
print(mask_pred, '-> mask_pred')

# convert masks to boolean, with a threshold of 0.5
mask_true = tf.cast(mask_true, tf.bool)
print(mask_true, '-> mask_true')
mask_pred = tf.cast(tf.where(
    tf.less(mask_pred, tf.zeros_like(mask_pred) + 0.5),
    tf.zeros_like(mask_pred),
    tf.ones_like(mask_pred)
), tf.bool)
print(mask_pred, '-> mask_pred')


print('\n===============================================================\n')
# Dice
intersection = tf.count_nonzero(tf.logical_and(mask_true, mask_pred),
                                [1, 2])  # [N]
print(intersection, '-> intersection')
union = tf.count_nonzero(mask_true, [1, 2]) + tf.count_nonzero(mask_pred,
                                                               [1, 2])  # [N]
print(union, '-> union')
intersection = tf.cast(intersection, tf.float32)
print(intersection, '-> intersection')
union = tf.cast(union, tf.float32)
print(union, '-> union')
coef = tf.math.divide(tf.math.add(2 * intersection, smooth),
                      tf.math.add(union, smooth))


print('\n===============================================================\n')
gt_bdry_score = coef
print(gt_bdry_score, '-> gt_bdry_score')
pred_bdry_score = tf.gather_nd(pred_bdry_score, indices)
print(pred_bdry_score, '-> pred_bdry_score')
print('Fix the shape:')
pred_bdry_score.set_shape((150,))
print(pred_bdry_score, '-> pred_bdry_score')

y_true = K.reshape(gt_bdry_score, (-1,))
print(y_true, '-> y_true')
y_pred = K.reshape(pred_bdry_score, (-1,))
print(y_pred, '-> y_pred')

loss = K.switch(tf.size(y_true) > 0,
                smooth_l1_loss(y_true, y_pred),
                tf.constant(0.0))
print(loss, '-> loss')
loss = K.mean(loss)