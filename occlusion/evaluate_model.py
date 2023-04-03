#%%
import time
import random
import matplotlib

from prepare_evaluation import *

ROOT_DIR = os.path.abspath("../")
sys.path.insert(0, ROOT_DIR)
from mrcnn import visualize
import mrcnn.model as modellib

MODEL_PATH = r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_012_bdry_120_m351\mask_rcnn_occlusion_0106.h5"
bdry = True
model_with_bdry, _, _ = prepare(MODEL_PATH, bdry)
MODEL_PATH = r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_004_120_m351\mask_rcnn_occlusion_0120.h5"
bdry = False
model_wo_bdry, config, dataset = prepare(MODEL_PATH, bdry)
matplotlib.use('TkAgg')
#%%
start = time.time()
plt.close('all')

target_ids = [
    "aeroplaneFGL1_BGL1/n02690373_3378",
    "busFGL1_BGL1/n02924116_600",
    "bottleFGL1_BGL1/n02823428_56",
    "busFGL1_BGL1/n02924116_11237",
    "trainFGL1_BGL1/n02917067_1701"
    ]

image_ids = search_image_ids(dataset, target_ids)

for image_id in image_ids:
    _, axes = plt.subplots(1, 3, figsize=(16, 6))
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                           dataset.image_reference(image_id)))

    results1 = model_with_bdry.detect([image], verbose=0)
    results2 = model_wo_bdry.detect([image], verbose=0)

    r = results1[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], ax=axes[0],
                                title="modified")

    r = results2[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], ax=axes[1],
                                title="original")

    visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id,
                                dataset.class_names, ax=axes[2],
                                title="GT")

    plt.subplots_adjust(left=0, right=1, bottom=0, wspace=0)
    plt.show()

end = time.time()

print("Time taken: ", end - start)
