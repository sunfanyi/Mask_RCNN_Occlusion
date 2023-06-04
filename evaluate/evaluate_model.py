#%%
# import time
# import random

import matplotlib
matplotlib.use('TkAgg')

from prepare_evaluation import *
from tools_evaluation import *
from calc_bdry_score_tf import run_graph

ROOT_DIR = os.path.abspath("../")
# ROOT_DIR = r"/rds/general/user/fs1519/home/FYP/Mask_RCNN-Occlusion"
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn import utils


#%% Model paths

dataset = 'surgical'
# dataset = 'occlusion'

# ======================= test normalisation & epochs settings =======================
to_test = 'epochs'
# epochs_settings = 0  # all
epochs_settings = 1  # 24
# epochs_settings = 2  # 100
# epochs_settings = 3  # 120 (1)
# epochs_settings = 4  # 120 (2)
all_model_paths, model_names = choose_setting(dataset, to_test, epochs_setting=epochs_settings)

# ======================= test boundary head input =======================
# to_test = 'bdry_input'
# norm = 'all'
# norm = 'max'
# norm = 'area'
# all_model_paths, model_names = choose_setting(dataset, to_test, norm=norm)

# ======================= test backbone =======================
# to_test = 'backbone'
# norm = 'all'
# norm = 'raw'
# norm = 'max'
# norm = 'area'
# all_model_paths, model_names = choose_setting(dataset, to_test, norm=norm)

# all_model_paths = [r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_019_sur_120_raw\mask_rcnn_surgical_0120.h5"]
# model_names = ['034_e120_bdry']

# all_model_paths = [all_model_paths[:3]]
# model_names = [model_names[:3]]

num_models = len(all_model_paths)

#%% Load dataset
subset = 'test'
# subset = 'train'

config, dataset = prepare_dataset_config(dataset, subset)

#%% Load weights
all_models = []
for path in all_model_paths:
    _config = config
    if to_test == 'backbone':
        if 'resnet50' in path:
            _config.BACKBONE = "resnet50"
        else:
            _config.BACKBONE = "resnet101"
    model = prepare_model(path, _config)
    all_models.append(model)

#%% Prediction
# start = time.time()

# img_names = [
#     "aeroplaneFGL1_BGL1/n02690373_3378",
#     "busFGL1_BGL1/n02924116_600",
#     "bottleFGL1_BGL1/n02823428_56",
#     "busFGL1_BGL1/n02924116_11237",
#     "trainFGL1_BGL1/n02917067_1701"
#     ]
# img_ids = search_image_ids(dataset, img_names)

# img_ids = dataset.image_ids
# img_ids = np.random.choice(dataset.image_ids, 5, replace=False)
img_ids = [9]
img_names = [dataset.image_info[i]['id'] for i in img_ids]

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

# end = time.time()
#
# print("Time taken: ", end - start)

#%% Evaluation

# img_ids = [30]
# img_names = [dataset.image_info[i]['id'] for i in img_ids]

df_iou = pd.DataFrame(columns=['id (iou)', 'images']+model_names)
df_dice = pd.DataFrame(columns=['id (dice)', 'images']+model_names)
df_ap50 = pd.DataFrame(columns=['id (ap50)', 'images']+model_names)
df_ap75 = pd.DataFrame(columns=['id (ap75)', 'images']+model_names)
df_ap90 = pd.DataFrame(columns=['id (ap90)', 'images']+model_names)
df_ap = pd.DataFrame(columns=['id (ap)', 'images']+model_names)

all_gt_masks = []
all_pred_masks = {i: [] for i in model_names}
all_bboxes = []


for i in range(len(img_ids)):
    id = img_ids[i]
    # print(id)
    name = img_names[i]

    gt = all_gt[i]
    all_gt_masks.extend([gt['masks'][:, :, _]
                        for _ in range(gt['masks'].shape[-1])])
    num_ROIs = gt['rois'].shape[0]
    all_bboxes.extend([gt['rois'][_] for _ in range(num_ROIs)])

    ious = []  # each element is a list of ious for each model
    dices = []  # each element is a list of ious for each model
    aps = []  # each element is a value of ap for each model
    aps50 = []  # each element is a value of ap for each model
    aps75 = []  # each element is a value of ap for each model
    aps90 = []  # each element is a value of ap for each model
    for model_name in model_names:
        r = all_predictions[model_name][i]
        # at 50 IOU:
        AP50, precisions50, recalls50, iou, dice, gt_match50 = compute_ap(
            gt['rois'], gt['class_ids'], gt['masks'],
            r['rois'], r['class_ids'], r['scores'], r['masks'])

        # at 75 IOU:
        AP75, precisions75, recalls75, _, _, _ = compute_ap(
            gt['rois'], gt['class_ids'], gt['masks'],
            r['rois'], r['class_ids'], r['scores'], r['masks'], iou_threshold=0.75)

        # at 90 IOU:
        AP90, precisions90, recalls90, _, _, _ = compute_ap(
            gt['rois'], gt['class_ids'], gt['masks'],
            r['rois'], r['class_ids'], r['scores'], r['masks'], iou_threshold=0.90)

        AP_range = compute_ap_range(
            gt['rois'], gt['class_ids'], gt['masks'],
            r['rois'], r['class_ids'], r['scores'], r['masks'], verbose=None)

        iou = get_matched_iou(iou, gt_match50)
        dice = get_matched_iou(dice, gt_match50)
        masks = get_matched_mask(r['masks'], gt_match50)

        ious.append(iou)
        dices.append(dice)
        aps.append(AP_range)
        aps50.append(AP50)
        aps75.append(AP75)
        aps90.append(AP90)

        all_pred_masks[model_name].extend([masks[_, :, :]
                                          for _ in range(masks.shape[0])])

    df_ap.loc[len(df_ap)] = [id, name] + aps
    df_ap50.loc[len(df_ap50)] = [id, name] + aps50
    df_ap75.loc[len(df_ap75)] = [id, name] + aps75
    df_ap90.loc[len(df_ap90)] = [id, name] + aps90

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
def get_averaged_df(model_names, df_ap, df_ap50, df_ap75,
                    df_ap90, df_bdry, df_iou, df_dice, fill_nan=True):
    num_images = df_ap.shape[0]
    num_rois = df_iou.shape[0]
    print("Number of images: ", num_images)
    print("Number of ROIs: ", num_rois)
    index = pd.MultiIndex.from_tuples([(x, y)
                                       for x in ['mu', 'std', 'delta_mu']
                                       for y in ['AP', 'AP50', 'AP75', 'AP90', 'bdry', 'iou', 'dice']])
    data = []
    all_mu = []
    all_std = []
    all_delta_mu = []

    # for map:
    for df in (df_ap, df_ap50, df_ap75, df_ap90):
        std = df.iloc[:, 2:].std(skipna=True)
        mu = df.iloc[:, 2:].mean(skipna=True)
        delta_mu = std / np.sqrt(num_images)
        all_mu.append(mu)
        all_std.append(std)
        all_delta_mu.append(delta_mu)

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
        all_mu.append(mu)
        all_std.append(std)
        all_delta_mu.append(delta_mu)

    data = all_mu + all_std + all_delta_mu
    data = np.array(data).T

    df_summary = pd.DataFrame(data, columns=index)
    df_summary.index = model_names
    df_summary.to_csv(sys.stdout, sep='\t', index=True)
    return df_summary


_averaged_df_dummy = get_averaged_df(model_names, df_ap, df_ap50,
                                     df_ap75, df_ap90, df_bdry, df_iou, df_dice, fill_nan=False)
_averaged_df = get_averaged_df(model_names, df_ap, df_ap50,
                               df_ap75, df_ap90, df_bdry, df_iou, df_dice, fill_nan=True)


#%% all data

metrics = ['AP', 'AP50', 'AP75', 'AP90', 'bdry', 'iou', 'dice']
_all_dfs = []

for idx, df in enumerate([df_ap, df_ap50, df_ap75, df_ap90, df_bdry, df_iou, df_dice]):
    _df = df.rename(columns={df.columns[0]: 'id'})
    _df = _df.melt(id_vars=['id', 'images'],
                   var_name='model_name',
                   value_name=metrics[idx])
    _all_dfs.append(_df)

# df_final = reduce(lambda left,right: pd.merge(left,right,on=['id', 'images', 'model_name']), _all_dfs)
df_final = _all_dfs[0]

for df in _all_dfs[1:]:
    df_final = pd.merge(df_final, df, on=['id', 'images', 'model_name'])

df_final.to_csv('all_data.csv', index=False)
#%% Visualize

from mrcnn import visualize

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
# ids_to_show = np.random.choice(img_ids, 5, replace=False)
ids_to_show = [9]

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
        score_to_show = r['scores']
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






