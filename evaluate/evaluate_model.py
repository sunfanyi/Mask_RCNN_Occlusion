#%%
import time
import random
import matplotlib

from prepare_evaluation import *
from tools_evaluation import *
from calc_bdry_score_tf import run_graph

ROOT_DIR = os.path.abspath("../")
sys.path.insert(0, ROOT_DIR)
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import utils

subset = 'test'

all_model_paths = [
    # r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\_train_019_sur_raw_120\mask_rcnn_surgical_0120.h5",
    # r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\_train_020_sur_bdry_120\mask_rcnn_surgical_0120.h5",
    r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\_train_025_occ_raw_120_m351\mask_rcnn_occlusion_0120.h5",
    r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\_train_026_occ_bdry_120_m351\mask_rcnn_occlusion_0120.h5",
]
model_names = [
    '015_original_100',
    '016_modified_100',
]
dataset = 'surgical' if 'sur' in all_model_paths[0] else 'occluded'
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

img_ids = np.random.choice(dataset.image_ids, 1)

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


end = time.time()

print("Time taken: ", end - start)


#%% Evaluation

img_names = [dataset.image_info[i]['id'] for i in img_ids]

# img_ids = [30]
# img_names = [dataset.image_info[i]['id'] for i in img_ids]

df_iou = pd.DataFrame(columns=['id (iou)', 'images']+model_names)
df_dice = pd.DataFrame(columns=['id (dice)', 'images']+model_names)
df_ap = pd.DataFrame(columns=['id (ap)', 'images']+model_names)

all_gt_masks = []
all_pred_masks = {model_names[i]: [] for i in range(num_models)}


def get_matched_mask(mask, gt_match):
    gt_match = gt_match.astype(np.int32)
    matched_iou = np.ndarray([len(gt_match), mask.shape[0], mask.shape[1]], dtype=np.bool)
    for i in range(len(gt_match)):
        match_iou = mask[:, :, gt_match[i]] if \
            gt_match[i] != -1 else np.zeros_like(mask[:, :, 0], dtype=np.bool)
        matched_iou[i] = match_iou
    return matched_iou


for i in range(len(img_ids)):
    id = img_ids[i]
    name = img_names[i]

    gt = all_gt[i]
    all_gt_masks.extend(gt['masks'][:, :, i]
                        for i in range(gt['masks'].shape[-1]))
    num_ROIs = gt['rois'].shape[0]

    ious = []  # each element is a list of ious for each model
    dices = []  # each element is a list of ious for each model
    aps = []  # each element is a value of ap for each model
    for model_name in model_names:
        r = all_predictions[model_name][i]
        gt_match, pred_match, iou, dice = compute_matches(
            gt['rois'], gt['class_ids'], gt['masks'],
            r['rois'], r['class_ids'], r['scores'], r['masks'])
        AP, precisions, recalls = compute_ap(gt_match, pred_match)
        iou = get_matched_iou(iou, gt_match)
        dice = get_matched_iou(dice, gt_match)
        masks = get_matched_mask(r['masks'], gt_match)

        ious.append(iou)
        dices.append(dice)
        aps.append(AP)

        all_pred_masks[model_name].extend(masks[j, :, :]
                                          for j in range(masks.shape[0]))

    df_ap.loc[len(df_ap)] = [id, name] + aps

    ious = np.array(ious)
    dices = np.array(dices)
    for roi in range(num_ROIs):
        df_iou.loc[len(df_iou)] = [id, name] + ious[:, roi].tolist()
        df_dice.loc[len(df_dice)] = [id, name] + dices[:, roi].tolist()

# Compute bdry_score
bdrys = []
for i in range(num_models):
    bdry = run_graph(all_gt_masks, all_pred_masks[model_names[i]])[0]
    bdrys.append(bdry)

bdrys = np.array(bdrys)
df_bdry = df_iou.copy()
df_bdry.iloc[:, 2:] = bdrys

df_iou = get_averaged_df(df_iou)
df_dice = get_averaged_df(df_dice)
df_ap = get_averaged_df(df_ap)
df_bdry = get_averaged_df(df_bdry)

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
ids_to_show = np.random.choice(img_ids, 1, replace=False)
# ids_to_show = [317]
names_to_show = [dataset.image_info[i]['id'] for i in ids_to_show]

plt.close('all')
for i in range(len(ids_to_show)):
    id = ids_to_show[i]
    name = names_to_show[i]

    fig, axes = get_ax(num_models+1)
    gt = all_gt[i]

    for j in range(num_models):
        r = all_predictions[model_names[j]][i]
        gt_match, pred_match, iou = utils.compute_matches(
            gt['rois'], gt['class_ids'], gt['masks'],
            r['rois'], r['class_ids'], r['scores'], r['masks'])
        iou = get_matched_iou(iou, gt_match)
        score_to_show = [iou[_] if _ != -1 else np.nan
                            for _ in pred_match.astype(np.int32)]
        visualize.display_instances(all_imgs[i], r['rois'], r['masks'], r['class_ids'],
                                    dataset.class_names, score_to_show, ax=axes[j],
                                    title=model_names[j])

    # ground truth
    visualize.display_instances(all_imgs[i], gt['rois'], gt['masks'], gt['class_ids'],
                                dataset.class_names, ax=axes[-1],
                                title="GT")

    # fig.tight_layout()
    fig.suptitle(name)
    plt.subplots_adjust(left=0, right=1, bottom=0, wspace=0)
    plt.show()


