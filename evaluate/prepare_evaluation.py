import os
import sys
import math
import tensorflow as tf
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("../")
sys.path.insert(0, ROOT_DIR)

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

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
    return fig, axes


def prepare_dataset_config(dataset, subset):
    if dataset == 'occluded':
        from occlusion import occlusion
        config = occlusion.OcclusionConfig()
        dataset = occlusion.OcclusionDataset()
        dataset_DIR = '../../datasets/dataset_occluded'
        dataset.load_occlusion(dataset_DIR, subset)
    elif dataset == 'surgical':
        from surgical_data import surgical
        config = surgical.SurgicalConfig()
        dataset = surgical.SurgicalDataset()
        dataset_DIR = '../../datasets/3dStool'
        dataset.load_surgical(dataset_DIR, subset)
    else:
        raise Exception('dataset not found')

    class InferenceConfig(config.__class__):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()

    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    return config, dataset


def prepare_model(MODEL_PATH, config, bdry):
    if bdry:
        import mrcnn.model as modellib
    else:
        import mrcnn.model_wo_bdry as modellib

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)

    print("Loading weights ", MODEL_PATH)
    model.load_weights(MODEL_PATH, by_name=True)
    return model


if __name__ == '__main__':
    MODEL_PATH = r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_012_bdry_120_m351\mask_rcnn_occlusion_0106.h5"

