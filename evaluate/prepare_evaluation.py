import os
import sys
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)  # To find local version of the library

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# env = 'local'
env = 'hpc'

if env == 'local':
    DEFAULT_LOGS_DIR = r"../logs"
    my_LOGS_DIR = r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs"
    sur_dataset_DIR = r'../../datasets/dataset_occluded'
    occ_dataset_DIR = r'../../datasets/3dStool'
elif env == 'hpc':
    DEFAULT_LOGS_DIR = r"/rds/general/user/fs1519/home/FYP/Mask_RCNN-Occlusion/logs"
    my_LOGS_DIR = r"/rds/general/user/fs1519/home/FYP/Mask_RCNN-Occlusion/my_logs"
    sur_dataset_DIR = r"/rds/general/user/fs1519/home/FYP/datasets/dataset_occluded"
    occ_dataset_DIR = r"/rds/general/user/fs1519/home/FYP/datasets/3dStool"


def choose_setting(dataset, to_test, epochs_setting=None, norm=None):
    assert dataset in ['surgical', 'occlusion']

    if to_test == 'epochs':
        def _get_epochs_24():
            h5_name = 'mask_rcnn_{}_0024.h5'.format(dataset)
            if dataset == 'surgical':
                model_names = [
                    'train_017_sur_24_raw',
                    'train_018_sur_24_max',
                    'train_028_sur_24_area',
                ]
            elif dataset == 'occlusion':
                model_names = [
                    'train_023_occ_24_raw',
                    'train_024_occ_24_max',
                    'train_031_occ_24_area',
                ]
            return h5_name, model_names

        def _get_epochs_100():
            h5_name = 'mask_rcnn_{}_0100.h5'.format(dataset)
            if dataset == 'surgical':
                model_names = [
                    'train_015_sur_100_raw',
                    'train_016_sur_100_max',
                    'train_027_sur_100_area',
                ]
            elif dataset == 'occlusion':
                model_names = [
                    'train_021_occ_100_raw',
                    'train_022_occ_100_max',
                    'train_030_occ_100_area',
                ]
            return h5_name, model_names

        def _get_epochs_120_1():
            h5_name = 'mask_rcnn_{}_0120.h5'.format(dataset)
            if dataset == 'surgical':
                model_names = [
                    'train_019_sur_120_raw',
                    'train_020_sur_120_max',
                    'train_029_sur_120_area',
                ]
            elif dataset == 'occlusion':
                model_names = [
                    'train_025_occ_120_raw',
                    'train_026_occ_120_max',
                    'train_032_occ_120_area',
                ]
            return h5_name, model_names

        def _get_epochs_120_2():
            h5_name = 'mask_rcnn_{}_0120.h5'.format(dataset)
            if dataset == 'surgical':
                model_names = [
                    'train_036_sur_120new_raw',
                    'train_035_sur_120new_max',
                    'train_033_sur_120new_area',
                ]
            elif dataset == 'occlusion':
                model_names = [
                    'train_038_occ_120new_raw',
                    'train_037_occ_120new_max',
                    'train_034_occ_120new_area',
                ]
            return h5_name, model_names

        h5_name, name1 = _get_epochs_24()
        path1 = [os.path.join(my_LOGS_DIR, i, h5_name) for i in name1]
        h5_name, name2 = _get_epochs_100()
        path2 = [os.path.join(my_LOGS_DIR, i, h5_name) for i in name2]
        h5_name, name3 = _get_epochs_120_1()
        path3 = [os.path.join(my_LOGS_DIR, i, h5_name) for i in name3]
        h5_name, name4 = _get_epochs_120_2()
        path4 = [os.path.join(my_LOGS_DIR, i, h5_name) for i in name4]
        if epochs_setting == 0:
            model_names = name1 + name2 + name3 + name4
            all_model_paths = path1 + path2 + path3 + path4
        else:
            model_names = eval('name{}'.format(epochs_setting))
            all_model_paths = eval('path{}'.format(epochs_setting))
    elif to_test == 'bdry_input':
        sub_DIR = 'trains_boundary_head_input'
        h5_name = 'mask_rcnn_{}_0024.h5'.format(dataset)

        def _get_norm_max():
            if dataset == 'surgical':
                model_names = ['train_0{}_sur_max_input{}'.
                               format(i+39, i+1) for i in range(9)]
            elif dataset == 'occlusion':
                model_names = ['train_0{}_occ_max_input{}'.
                               format(i+57, i+1) for i in range(9)]
            return model_names

        def _get_norm_area():
            if dataset == 'surgical':
                model_names = ['train_0{}_sur_area_input{}'.
                               format(i+48, i+1) for i in range(9)]
            elif dataset == 'occlusion':
                model_names = ['train_0{}_occ_area_input{}'.
                               format(i+66, i+1) for i in range(9)]
            return model_names

        name_max = _get_norm_max()
        path_max = [os.path.join(my_LOGS_DIR, sub_DIR, i, h5_name) for i in name_max]
        name_area = _get_norm_area()
        path_area = [os.path.join(my_LOGS_DIR, sub_DIR, i, h5_name) for i in name_area]
        if norm == 'all':
            model_names = name_max + name_area
            all_model_paths = path_max + path_area
        else:
            model_names = eval('name_{}'.format(norm))
            all_model_paths = eval('path_{}'.format(norm))
    elif to_test == 'backbone':
        sub_DIR = 'trains_backbone'
        h5_name = 'mask_rcnn_{}_0024.h5'.format(dataset)

        def _get_norm_raw():
            if dataset == 'surgical':
                model_names = ['train_017_sur_24_raw',
                               'train_075_sur_resnet50_raw']
            elif dataset == 'occlusion':
                model_names = ['train_023_occ_24_raw',
                               'train_078_occ_resnet50_raw']
            return model_names

        def _get_norm_max():
            if dataset == 'surgical':
                model_names = ['train_018_sur_24_max',
                               'train_076_sur_resnet50_max']
            elif dataset == 'occlusion':
                model_names = ['train_024_occ_24_max',
                               'train_079_occ_resnet50_max']
            return model_names

        def _get_norm_area():
            if dataset == 'surgical':
                model_names = ['train_028_sur_24_area',
                               'train_077_sur_resnet50_area']
            elif dataset == 'occlusion':
                model_names = ['train_031_occ_24_area',
                               'train_080_occ_resnet50_area']
            return model_names

        name_raw = _get_norm_raw()
        path_raw = [os.path.join(my_LOGS_DIR, sub_DIR, i, h5_name) for i in name_raw]
        name_max = _get_norm_max()
        path_max = [os.path.join(my_LOGS_DIR, sub_DIR, i, h5_name) for i in name_max]
        name_area = _get_norm_area()
        path_area = [os.path.join(my_LOGS_DIR, sub_DIR, i, h5_name) for i in name_area]
        if norm == 'all':
            model_names = name_raw + name_max + name_area
            all_model_paths = path_raw + path_max + path_area
        else:
            model_names = eval('name_{}'.format(norm))
            all_model_paths = eval('path_{}'.format(norm))
    else:
        raise Exception('to_test not found')
    model_names = [i.replace('train_', '') for i in model_names]
    return all_model_paths, model_names


def prepare_dataset_config(dataset, subset):
    if dataset == 'occlusion':
        from occlusion import occlusion
        config = occlusion.OcclusionConfig()
        dataset = occlusion.OcclusionDataset()
        dataset.load_occlusion(sur_dataset_DIR, subset)
    elif dataset == 'surgical':
        from surgical_data import surgical
        config = surgical.SurgicalConfig()
        dataset = surgical.SurgicalDataset()
        dataset.load_surgical(occ_dataset_DIR, subset)
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
    if 'raw' in MODEL_PATH:
        import mrcnn.model_wo_bdry as modellib
    else:
        import mrcnn.model as modellib

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=DEFAULT_LOGS_DIR,
                                  config=config)

    print("Loading weights ", MODEL_PATH)
    model.load_weights(MODEL_PATH, by_name=True)
    return model


if __name__ == '__main__':
    MODEL_PATH = r"D:\Users\ROG\Desktop\FYP\Mask_RCNN-Occulusion\logs\train_012_bdry_120_m351\mask_rcnn_occlusion_0106.h5"

