#%%
import time
import random
import pandas as pd
import numpy as np
import matplotlib

from prepare_evaluation import *

ROOT_DIR = os.path.abspath("../")
sys.path.insert(0, ROOT_DIR)
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import utils

subset = 'test'
dataset = 'occluded'
# dataset = 'surgical'

all_model_paths = [
    r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\_train_021_occ_raw_100_m351\mask_rcnn_occlusion_0100.h5",
    # r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\_train_020_sur_bdry_120\mask_rcnn_surgical_0120.h5",
]
model_names = [
    '015_original_100',
    # '016_modified_100',
]
num_models = len(all_model_paths)

config, dataset = prepare_dataset_config(dataset, subset)

all_models = []
for path in all_model_paths:
    if 'bdry' in path:
        model = prepare_model(path, config, True)
    else:
        model = prepare_model(path, config, False)
    all_models.append(model)

#%% Prediction
start = time.time()

# img_names = [
#     "aeroplaneFGL1_BGL1/n02690373_3378",
#     "busFGL1_BGL1/n02924116_600",
#     "bottleFGL1_BGL1/n02823428_56",
#     "busFGL1_BGL1/n02924116_11237",
#     "trainFGL1_BGL1/n02917067_1701"
#     ]
# img_ids = search_image_ids(dataset, img_names)

img_ids = dataset.image_ids[:10]

all_imgs = []
all_gt = []
all_predictions = {model_names[i]: [] for i in range(num_models)}
for image_id in img_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                           dataset.image_reference(image_id)))
    all_imgs.append(image)
    all_gt.append({'class_ids': gt_class_id, 'rois': gt_bbox, 'masks': gt_mask})

    for i in range(num_models):
        _detected = all_models[i].detect([image], verbose=0)[0]
        all_predictions[model_names[i]].append(_detected)
    # # detect first 10 images
    # if len(all_predictions[model_names[0]]) == 10:
    #     break


end = time.time()

print("Time taken: ", end - start)


#%% Evaluation

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

    dice = (2. * intersections + smooth) / (area1 + area2 + smooth)

    return dice


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
    gt_match, pred_match, overlaps = compute_matches(
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

    if return_match:
        return mAP, precisions, recalls, overlaps, gt_match
    else:
        return mAP, precisions, recalls, overlaps


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
    # overlaps = compute_overlaps_masks(pred_masks, gt_masks)
    overlaps = calculate_dice_coefficient(pred_masks, gt_masks)

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

    return gt_match, pred_match, overlaps




def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


img_names = [dataset.image_info[i]['id'] for i in img_ids]

# img_ids = [30]
# img_names = [dataset.image_info[i]['id'] for i in img_ids]


def get_matched_iou(iou, gt_match):
    gt_match = gt_match.astype(np.int32)
    matched_iou = np.ndarray([len(gt_match)], dtype=np.float32)
    for i in range(len(gt_match)):
        match_iou = iou[gt_match[i], i] if gt_match[i] != -1 else np.nan
        matched_iou[i] = match_iou
    return matched_iou


df_iou = pd.DataFrame(columns=['id (iou)', 'images']+model_names)
df_ap = pd.DataFrame(columns=['id (ap)', 'images']+model_names)

for id, name in zip(img_ids, img_names):
    gt = all_gt[id]

    ious = []  # each element is a list of ious for each model
    aps = []  # each element is a value of ap for each model
    for i in range(num_models):
        r = all_predictions[model_names[i]][id]
        # gt_match, _, iou = utils.compute_matches(
        #     gt['rois'], gt['class_ids'], gt['masks'],
        #     r['rois'], r['class_ids'], r['scores'], r['masks'])
        AP, precisions, recalls, iou, gt_match =\
            compute_ap(gt['rois'], gt['class_ids'], gt['masks'],
                             r['rois'], r['class_ids'], r['scores'], r['masks'],
                             return_match=True)
        iou = get_matched_iou(iou, gt_match)
        ious.append(iou)
        aps.append(AP)

    num_ROIs = gt['rois'].shape[0]
    print(num_ROIs)
    iou_data = [[id, name] + [ious[i][j] for i in range(num_models)]
                for j in range(num_ROIs)]
    ap_data = [id, name] + [aps[i] for i in range(num_models)]

    df_ap.loc[len(df_ap)] = ap_data

    for iou in iou_data:
        df_iou.loc[len(df_iou)] = iou










def get_averaged_df(df):
    avg = df.iloc[:, 2:].mean(skipna=True)
    # avg = df.fillna(0).mean(skipna=True)
    avg_df = pd.DataFrame(columns=df.columns)
    avg_df.loc[0] = ['average', 'average'] + avg.tolist()

    averaged_df = pd.concat([avg_df, df]).reset_index(drop=True)
    averaged_df.to_csv(sys.stdout, sep='\t', index=False)
    return averaged_df


df_iou = get_averaged_df(df_iou)
df_ap = get_averaged_df(df_ap)

#%% Visualize
matplotlib.use('TkAgg')
# names_to_show = [  # in test set
#     "bottleFGL1_BGL1/n02823428_1219",
#     'trainFGL1_BGL1/n02917067_8822',
#     'bottleFGL1_BGL1/n02823428_957',
#     'aeroplaneFGL1_BGL1/n02690373_2091',
#     'bottleFGL1_BGL1/n02823428_436',
#     'busFGL1_BGL1/n02924116_11237',
#     'busFGL1_BGL1/n02924116_52499'
#     ]
# ids_to_show = search_image_ids(dataset, names_to_show)

# random 5 images
# ids_to_show = np.random.choice(img_ids, 5)
ids_to_show = [0, 1, 2]
names_to_show = [dataset.image_info[i]['id'] for i in ids_to_show]

plt.close('all')
for id, name in zip(ids_to_show, names_to_show):
    fig, axes = get_ax(num_models+1)
    gt = all_gt[id]

    for i in range(num_models):
        r = all_predictions[model_names[i]][id]
        gt_match, pred_match, iou = utils.compute_matches(
            gt['rois'], gt['class_ids'], gt['masks'],
            r['rois'], r['class_ids'], r['scores'], r['masks'])
        iou = get_matched_iou(iou, gt_match)
        score_to_show = [iou[_] if _ != -1 else np.nan
                            for _ in pred_match.astype(np.int32)]
        visualize.display_instances(all_imgs[id], r['rois'], r['masks'], r['class_ids'],
                                    dataset.class_names, score_to_show, ax=axes[i],
                                    title=model_names[i])

    # ground truth
    visualize.display_instances(all_imgs[id], gt['rois'], gt['masks'], gt['class_ids'],
                                dataset.class_names, ax=axes[-1],
                                title="GT")

    # fig.tight_layout()
    fig.suptitle(name)
    plt.subplots_adjust(left=0, right=1, bottom=0, wspace=0)
    plt.show()


