#%%
import os
import sys
import time
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from prepare_evaluation import get_ax, prepare

ROOT_DIR = os.path.abspath("../")
sys.path.insert(0, ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

MODEL_PATH = r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_012_bdry_120_m351\mask_rcnn_occlusion_0106.h5"
model, config, dataset = prepare(MODEL_PATH)
matplotlib.use('TkAgg')
#%%
start = time.time()
plt.close('all')

# image_id = random.choice(dataset.image_ids)
for image_id in [10, 11, 12, 13]:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                           dataset.image_reference(image_id)))
    results = model.detect([image], verbose=1)

    # Display results
    ax = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], ax=ax,
                                title="Predictions")
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    plt.show()

end = time.time()

print("Time taken: ", end - start)
