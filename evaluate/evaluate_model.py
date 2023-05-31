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


#%% Model paths

# setting = 'sur 24 epochs'
# setting = 'occ 24 epochs'
# setting = 'sur 100 epochs'
# setting = 'occ 100 epochs'
# setting = 'sur 120 epochs'
setting = 'occ 120 epochs'

all_model_paths, model_names = choose_setting(setting)

# all_model_paths = [r"d:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_034_occ_120new_area\mask_rcnn_occlusion_0120.h5"]
# model_names = ['034_e120_bdry']
# setting = 'sur 24 epoches'
# p1, n1 = choose_setting(setting)
# setting = 'sur 100 epoches'
# p2, n2 = choose_setting(setting)
# all_model_paths = p1 + p2
# model_names = n1 + n2

# all_model_paths = [
#     r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\_train_040_sur_msrcnn_24\mask_rcnn_surgical_0024.h5",
# ]
# model_names = [
#     '040_e24_msrcnn',
# ]

num_models = len(all_model_paths)

#%% Load dataset
subset = 'test'
# subset = 'train'
dataset = 'surgical' if 'sur' in all_model_paths[0] else 'occluded'

config, dataset = prepare_dataset_config(dataset, subset)

#%% Load weights
all_models = []
for path in all_model_paths:
    model = prepare_model(path, config)
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

# img_ids = dataset.image_ids
img_ids = np.random.choice(dataset.image_ids, 5, replace=False)
# img_ids = [23]

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
all_pred_masks = {i: [] for i in model_names}
all_bboxes = []


for i in range(len(img_ids)):
    id = img_ids[i]
    print(id)
    name = img_names[i]

    gt = all_gt[i]
    all_gt_masks.extend([gt['masks'][:, :, _]
                        for _ in range(gt['masks'].shape[-1])])
    num_ROIs = gt['rois'].shape[0]
    all_bboxes.extend([gt['rois'][_] for _ in range(num_ROIs)])

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

        all_pred_masks[model_name].extend([masks[_, :, :]
                                          for _ in range(masks.shape[0])])

    df_ap.loc[len(df_ap)] = [id, name] + aps

    ious = np.array(ious)
    dices = np.array(dices)
    for roi in range(num_ROIs):
        df_iou.loc[len(df_iou)] = [id, name] + ious[:, roi].tolist()
        df_dice.loc[len(df_dice)] = [id, name] + dices[:, roi].tolist()


# Compute bdry_score
bdrys = []
for i in range(num_models):
    bdry = run_graph(all_gt_masks, all_pred_masks[model_names[i]], all_bboxes)[0]
    bdrys.append(bdry)

bdrys = np.array(bdrys)
df_bdry = df_iou.copy()
df_bdry.iloc[:, 2:] = bdrys.T


#%% Data processing
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


_averaged_df_dummy = get_averaged_df(model_names, df_ap, df_bdry, df_iou, df_dice, fill_nan=False)
_averaged_df = get_averaged_df(model_names, df_ap, df_bdry, df_iou, df_dice, fill_nan=True)

#%% Visualize
"""
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
ids_to_show = np.random.choice(img_ids, 5, replace=False)
# ids_to_show = [317]

plt.close('all')
for i in range(len(img_ids)):
    id = img_ids[i]
    if id not in ids_to_show:
        continue
    name = img_names[i]

    fig, axes = get_ax(num_models+1)
    axes = axes.flatten()
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


"""


