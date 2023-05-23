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

MODEL_PATH = r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_012_bdry_120_m351\mask_rcnn_occlusion_0106.h5"
bdry = True
model_with_bdry, _, _ = prepare(MODEL_PATH, bdry, subset)
MODEL_PATH = r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_004_120_m351\mask_rcnn_occlusion_0106.h5"
bdry = False
model_wo_bdry, config, dataset = prepare(MODEL_PATH, bdry, subset)
matplotlib.use('TkAgg')

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

img_ids = dataset.image_ids

all_imgs = []
all_gt = []
all_pred_modified = []
all_pred_original = []
for image_id in img_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                           dataset.image_reference(image_id)))
    all_imgs.append(image)
    all_gt.append({'class_ids': gt_class_id, 'rois': gt_bbox, 'masks': gt_mask})

    all_pred_modified.append(model_with_bdry.detect([image], verbose=0)[0])
    all_pred_original.append(model_wo_bdry.detect([image], verbose=0)[0])

    # # detect first 10 images
    # if len(all_pred_modified) % 10 == 0:
    #     break

end = time.time()

print("Time taken: ", end - start)


#%% Evaluation

img_ids = dataset.image_ids
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


df = pd.DataFrame(columns=['id', 'images', 'modified', 'original'])

for id, name in zip(img_ids, img_names):
    gt = all_gt[id]

    r = all_pred_modified[id]
    gt_match1, a, iou1 = utils.compute_matches(
        gt['rois'], gt['class_ids'], gt['masks'],
        r['rois'], r['class_ids'], r['class_scores'], r['masks'])
    iou_modified = get_matched_iou(iou1, gt_match1)

    r = all_pred_original[id]
    gt_match2, _, iou2 = utils.compute_matches(
        gt['rois'], gt['class_ids'], gt['masks'],
        r['rois'], r['class_ids'], r['scores'], r['masks'])
    iou_original = get_matched_iou(iou2, gt_match2)

    num_ROIs = gt['rois'].shape[0]
    # if num_ROIs > 1:
    #     continue
    data = [[id, name, iou_modified[j], iou_original[j]]
            for j in range(num_ROIs)]

    for row in data:
        df.loc[len(df)] = row


# delete wrong values
# cols = df.columns[-2:]
# df[cols] = df[cols].mask(df[cols] < 0.1, np.nan)

avg = df.mean(skipna=True)
print('average IOU overlap modified: ', avg[-2])
print('average IOU overlap original: ', avg[-1])
print('percentage improvement: ', (avg[-2] - avg[-1]) / avg[-1] * 100, '%')


#%% Visualize
names_to_show = [
    "bottleFGL1_BGL1/n02823428_1219",
    'trainFGL1_BGL1/n02917067_8822',
    'bottleFGL1_BGL1/n02823428_957',
    'aeroplaneFGL1_BGL1/n02690373_2091',
    'bottleFGL1_BGL1/n02823428_436',
    'busFGL1_BGL1/n02924116_11237',
    'busFGL1_BGL1/n02924116_52499'
    ]
ids_to_show = search_image_ids(dataset, names_to_show)

# # random 5 images
# # ids_to_show = np.random.choice(dataset.image_ids, 10)
# ids_to_show = [30]
# names_to_show = [dataset.image_info[i]['id'] for i in ids_to_show]

plt.close('all')
for id, name in zip(ids_to_show, names_to_show):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    gt = all_gt[id]

    # ======================== modified network ========================
    r = all_pred_modified[id]
    gt_match, pred_match, iou = utils.compute_matches(
        gt['rois'], gt['class_ids'], gt['masks'],
        r['rois'], r['class_ids'], r['class_scores'], r['masks'])
    iou_modified = get_matched_iou(iou, gt_match)
    # score_to_show = r['class_scores']
    score_to_show = [iou_modified[i] if i != -1 else np.nan
                     for i in pred_match.astype(np.int32)]
    visualize.display_instances(all_imgs[id], r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, score_to_show, ax=axes[0],
                                title="modified")

    # ======================== original network ========================
    r = all_pred_original[id]
    gt_match, pred_match, iou = utils.compute_matches(
        gt['rois'], gt['class_ids'], gt['masks'],
        r['rois'], r['class_ids'], r['scores'], r['masks'])
    iou_original = get_matched_iou(iou, gt_match)

    # score_to_show = r['scores']
    score_to_show = [iou_original[i] if i != -1 else np.nan
                     for i in pred_match.astype(np.int32)]
    visualize.display_instances(all_imgs[id], r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, score_to_show, ax=axes[1],
                                title="original")

    # ======================== ground truth ========================
    visualize.display_instances(all_imgs[id], gt['rois'], gt['masks'], gt['class_ids'],
                                dataset.class_names, ax=axes[2],
                                title="GT")

    fig.suptitle(name)
    plt.subplots_adjust(left=0, right=1, bottom=0, wspace=0)
    plt.show()

