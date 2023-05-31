import os
import sys
import tensorflow as tf

ROOT_DIR = os.path.abspath("../")
sys.path.insert(0, ROOT_DIR)

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


def choose_setting(setting):
    logDIR = r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs"

    if '24' in setting:
        if setting == 'sur 24 epochs':
            h5_name = 'mask_rcnn_surgical_0024.h5'
            model_names = [
                'train_017_sur_24_raw',
                'train_018_sur_24_max',
                'train_028_sur_24_area',
            ]
        elif setting == 'occ 24 epochs':
            h5_name = 'mask_rcnn_occlusion_0024.h5'
            model_names = [
                'train_023_occ_24_raw',
                'train_022_occ_24_max',
                'train_030_occ_24_area',
            ]
    elif '100' in setting:
        if setting == 'sur 100 epochs':
            h5_name = 'mask_rcnn_surgical_0100.h5'
            model_names = [
                'train_015_sur_100_raw',
                'train_016_sur_100_max',
                'train_027_sur_100_area',
            ]
        elif setting == 'occ 100 epochs':
            h5_name = 'mask_rcnn_occlusion_0100.h5'
            model_names = [
                'train_021_occ_100_raw',
                'train_024_occ_100_max',
                'train_031_occ_100_area',
            ]
    elif '120' in setting:
        if setting == 'sur 120 epochs':
            h5_name = 'mask_rcnn_surgical_0120.h5'
            model_names = [
                'train_019_sur_120_raw',
                'train_020_sur_120_max',
                'train_029_sur_120_area',
            ]
        elif setting == 'occ 120 epochs':
            h5_name = 'mask_rcnn_occlusion_0120.h5'
            model_names = [
                'train_025_occ_120_raw',
                'train_026_occ_120_max',
                'train_032_occ_120_area',
            ]
        # elif setting == 'sur new 120 epochs':
        # elif setting == 'occ new 120 epochs':
    else:
        raise ValueError('Invalid setting')

    all_model_paths = [os.path.join(logDIR, i, h5_name) for i in model_names]
    model_names = [i.replace('train_', '') for i in model_names]
    return all_model_paths, model_names


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


def prepare_model(MODEL_PATH, config):
    if 'bdry' in MODEL_PATH:
        import mrcnn.model as modellib
    elif 'msrcnn' in MODEL_PATH:
        import mrcnn.model_msrcnn as modellib
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

