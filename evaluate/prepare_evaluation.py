import os
import sys
import tensorflow as tf

ROOT_DIR = os.path.abspath("../")
sys.path.insert(0, ROOT_DIR)

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


def choose_setting(setting):
    if setting == 'sur 24 epoches':
        all_model_paths = [
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_017_sur_raw_24\mask_rcnn_surgical_0024.h5",
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_018_sur_bdry_24\mask_rcnn_surgical_0024.h5",
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_028_sur_bdry_24\mask_rcnn_surgical_0024.h5",
        ]
        model_names = [
            '017_e24_raw',
            '018_e24_norm_max',
            '028_e24_norm_roi',
        ]
    elif setting == 'occ 24 epoches':
        all_model_paths = [
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_023_occ_raw_24_m351\mask_rcnn_occlusion_0024.h5",
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_024_occ_bdry_24_m351\mask_rcnn_occlusion_0024.h5",
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_031_occ_bdry_24_m351\mask_rcnn_occlusion_0024.h5",
        ]
        model_names = [
            '023_e24_raw',
            '024_e24_norm_max',
            '031_e24_norm_roi',
        ]
    elif setting == 'sur 100 epoches':
        all_model_paths = [
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_015_sur_raw_100\mask_rcnn_surgical_0100.h5",
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_016_sur_bdry_100\mask_rcnn_surgical_0100.h5",
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_027_sur_bdry_100\mask_rcnn_surgical_0100.h5",
        ]
        model_names = [
            '015_e100_raw',
            '016_e100_norm_max',
            '027_e100_norm_roi',
        ]
    elif setting == 'occ 100 epoches':
        all_model_paths = [
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_021_occ_raw_100_m351\mask_rcnn_occlusion_0100.h5",
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_022_occ_bdry_100_m351\mask_rcnn_occlusion_0100.h5",
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_030_occ_bdry_100_m351\mask_rcnn_occlusion_0100.h5",
        ]
        model_names = [
            '021_e100_raw',
            '022_e100_norm_max',
            '030_e100_norm_roi',
        ]
    elif setting == 'sur 120 epoches':
        all_model_paths = [
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_019_sur_raw_120\mask_rcnn_surgical_0120.h5",
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_020_sur_bdry_120\mask_rcnn_surgical_0120.h5",
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_029_sur_bdry_120\mask_rcnn_surgical_0120.h5",
        ]
        model_names = [
            '019_e120_raw',
            '020_e120_norm_max',
            '029_e120_norm_roi',
        ]
    elif setting == 'occ 120 epoches':
        all_model_paths = [
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_025_occ_raw_120_m351\mask_rcnn_occlusion_0120.h5",
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_026_occ_bdry_120_m351\mask_rcnn_occlusion_0120.h5",
            r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_032_occ_bdry_120_m351\mask_rcnn_occlusion_0120.h5",
        ]
        model_names = [
            '025_e120_raw',
            '026_e120_norm_max',
            '032_e120_norm_roi',
        ]
    else:
        raise Exception('setting not found')
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

