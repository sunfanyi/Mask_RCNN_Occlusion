import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

import occlusion

ROOT_DIR = os.path.abspath("../")
sys.path.insert(0, ROOT_DIR)

config = occlusion.OcclusionConfig()
dataset_DIR = '../../datasets/dataset_occluded'


class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def search_image_ids(dataset, target_ids):
    image_ids = []
    for target_id in target_ids:
        for i in range(len(dataset.image_info)):
            if dataset.image_info[i]['id'] == target_id:
                image_ids.append(i)
                break
    return image_ids


def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def prepare(MODEL_PATH, bdry):
    if bdry:
        import mrcnn.model as modellib
    else:
        import mrcnn.model_wo_bdry as modellib

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    config = InferenceConfig()
    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

    dataset = occlusion.OcclusionDataset()
    dataset.load_occlusion(dataset_DIR, "test")
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)

    weights_path = MODEL_PATH

    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    return model, config, dataset


if __name__ == '__main__':
    MODEL_PATH = r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_012_bdry_120_m351\mask_rcnn_occlusion_0106.h5"
    model, config, dataset = prepare(MODEL_PATH, True)

