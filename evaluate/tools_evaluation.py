import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def search_image_ids(dataset, target_ids):
    image_ids = []
    for target_id in target_ids:
        source = 'occlusion.' + target_id
        image_ids.append(dataset.image_from_source_map[source])
        # for i in range(len(dataset.image_info)):
        #     if dataset.image_info[i]['id'] == target_id:
        #         image_ids.append(i)
        #         break
    return image_ids


def get_ax(num_images, size=6):
    if num_images <= 3:
        nrows, ncols = 1, num_images
    elif num_images <= 4:
        nrows, ncols = 2, 2
    elif num_images <= 6:
        nrows, ncols = 2, 3
    elif num_images <= 9:
        nrows, ncols = 3, 3
    else:
        nrows = math.ceil(num_images / 4)
        ncols = 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows))
    axes = axes.flatten()
    return fig, axes


def get_averaged_df(model_names, df_ap, df_bdry, df_iou, df_dice, fill_nan=True):
    num_images = df_ap.shape[0]
    num_rois = df_iou.shape[0]
    print("Number of images: ", num_images)
    print("Number of ROIs: ", num_rois)
    index = pd.MultiIndex.from_tuples([(x, y)
                                       for x in ['map', 'bdry', 'iou', 'dice']
                                       for y in ['mu', 'std', 'delta_mu']])
    data = []

    # for map:
    std = df_ap.iloc[:, 2:].std(skipna=True)
    mu = df_ap.iloc[:, 2:].mean(skipna=True)
    delta_mu = std / np.sqrt(num_images)
    data = data + [mu, std, delta_mu]

    for df in (df_bdry, df_iou, df_dice):
        # 0 indicates nothing detected, so we replace it with nan
        df = df.replace(0, np.nan)
        std = df_ap.iloc[:, 2:].std(skipna=True)
        if fill_nan:  # better
            df = df.fillna(0)
            mu = df.iloc[:, 2:].mean(skipna=False)
        else:
            mu = df.iloc[:, 2:].mean(skipna=True)
        delta_mu = std / np.sqrt(num_rois)
        data = data + [mu, std, delta_mu]

    data = np.array(data).T

    df_summary = pd.DataFrame(data, columns=index)
    df_summary.index = model_names
    df_summary.to_csv(sys.stdout, sep='\t', index=True)
    return df_summary


def get_matched_iou(iou, gt_match):
    gt_match = gt_match.astype(np.int32)
    matched_iou = np.ndarray([len(gt_match)], dtype=np.float32)
    for i in range(len(gt_match)):
        match_iou = iou[gt_match[i], i] if \
            gt_match[i] != -1 else np.nan
        matched_iou[i] = match_iou
    return matched_iou


def get_matched_mask(mask, gt_match):
    gt_match = gt_match.astype(np.int32)
    matched_masks = np.ndarray((len(gt_match), mask.shape[0], mask.shape[1]),
                               dtype=np.bool)
    for i in range(len(gt_match)):
        if gt_match[i] == -1:
            matched_masks[i] = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.bool)
        else:
            matched_masks[i] = mask[:, :, gt_match[i]]
    return matched_masks


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """

    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def calculate_dice_coefficient(masks1, masks2, smooth=0.00001):
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)

    dice = (2. * intersections + smooth) / (area1[:, None] + area2[None, :] + smooth)

    return dice


# def compute_ap_range(gt_box, gt_class_id, gt_mask,
#                      pred_box, pred_class_id, pred_score, pred_mask,
#                      iou_thresholds=None, verbose=1):
#     """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
#     # Default is 0.5 to 0.95 with increments of 0.05
#     iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
#
#     # Compute AP over range of IoU thresholds
#     AP = []
#     for iou_threshold in iou_thresholds:
#         ap, precisions, recalls, overlaps = \
#             compute_ap(gt_box, gt_class_id, gt_mask,
#                        pred_box, pred_class_id, pred_score, pred_mask,
#                        iou_threshold=iou_threshold)
#         if verbose:
#             print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
#         AP.append(ap)
#     AP = np.array(AP).mean()
#     if verbose:
#         print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
#             iou_thresholds[0], iou_thresholds[-1], AP))
#     return AP
#
#
# def compute_ap(gt_match, pred_match):
#     """Compute Average Precision at a set IoU threshold (default 0.5).
#
#     Returns:
#     mAP: Mean Average Precision
#     precisions: List of precisions at different class score thresholds.
#     recalls: List of recall values at different class score thresholds.
#     overlaps: [pred_boxes, gt_boxes] IoU overlaps.
#     """
#
#     # Compute precision and recall at each prediction box step
#     precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
#     recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)
#
#     # Pad with start and end values to simplify the math
#     precisions = np.concatenate([[0], precisions, [0]])
#     recalls = np.concatenate([[0], recalls, [1]])
#
#     # Ensure precision values decrease but don't increase. This way, the
#     # precision value at each recall threshold is the maximum it can be
#     # for all following recall thresholds, as specified by the VOC paper.
#     for i in range(len(precisions) - 2, -1, -1):
#         precisions[i] = np.maximum(precisions[i], precisions[i + 1])
#
#     # Compute mean AP over recall range
#     indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
#     mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
#                  precisions[indices])
#
#     return mAP, precisions, recalls
#
#
# def compute_matches(gt_boxes, gt_class_ids, gt_masks,
#                     pred_boxes, pred_class_ids, pred_scores, pred_masks,
#                     iou_threshold=0.5, score_threshold=0.0):
#     """Finds matches between prediction and ground truth instances.
#
#     Returns:
#         gt_match: 1-D array. For each GT box it has the index of the matched
#                   predicted box.
#         pred_match: 1-D array. For each predicted box, it has the index of
#                     the matched ground truth box.
#         overlaps: [pred_boxes, gt_boxes] IoU overlaps.
#     """
#     # Trim zero padding
#     # TODO: cleaner to do zero unpadding upstream
#     gt_boxes = trim_zeros(gt_boxes)
#     gt_masks = gt_masks[..., :gt_boxes.shape[0]]
#     pred_boxes = trim_zeros(pred_boxes)
#     pred_scores = pred_scores[:pred_boxes.shape[0]]
#     # Sort predictions by score from high to low
#     indices = np.argsort(pred_scores)[::-1]
#     pred_boxes = pred_boxes[indices]
#     pred_class_ids = pred_class_ids[indices]
#     pred_scores = pred_scores[indices]
#     pred_masks = pred_masks[..., indices]
#
#     # Compute IoU overlaps [pred_masks, gt_masks]
#     overlaps = compute_overlaps_masks(pred_masks, gt_masks)
#     dice = calculate_dice_coefficient(pred_masks, gt_masks)
#
#     # Loop through predictions and find matching ground truth boxes
#     match_count = 0
#     pred_match = -1 * np.ones([pred_boxes.shape[0]])
#     gt_match = -1 * np.ones([gt_boxes.shape[0]])
#
#     for i in range(len(pred_boxes)):
#         # Find best matching ground truth box
#         # 1. Sort matches by score
#         sorted_ixs = np.argsort(overlaps[i])[::-1]
#         # 2. Remove low scores
#         low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
#         if low_score_idx.size > 0:
#             sorted_ixs = sorted_ixs[:low_score_idx[0]]
#         # 3. Find the match
#         for j in sorted_ixs:
#             # If ground truth box is already matched, go to next one
#             if gt_match[j] > -1:
#                 continue
#             # If we reach IoU smaller than the threshold, end the loop
#             iou = overlaps[i, j]
#             if iou < iou_threshold:
#                 break
#             # Do we have a match?
#             if pred_class_ids[i] == gt_class_ids[j]:
#                 match_count += 1
#                 gt_match[j] = i
#                 pred_match[i] = j
#                 break
#
#     return gt_match, pred_match, overlaps, dice


def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)
    dice = calculate_dice_coefficient(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps, dice


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5, return_match=False):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps, dice = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps, dice, gt_match


def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps, _, _ = \
            compute_ap(gt_box, gt_class_id, gt_mask,
                       pred_box, pred_class_id, pred_score, pred_mask,
                       iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP


def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]
