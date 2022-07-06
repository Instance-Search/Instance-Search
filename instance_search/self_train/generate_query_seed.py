"""Test localizer with a pair of images.
Author: gongyou.zyq
Date: 2020.11.18
"""

import pickle
import os
import argparse
# from random import shuffle
import numpy as np
import cv2

from instance_search.utils.io_utils import merge_pred2group
from instance_search.config import cfg

np.random.seed(1)

def parse_args():
    """Parse args."""

    parser = argparse.ArgumentParser(description="Instance Search Config")
    parser.add_argument("--config_file", default="Cityflow.yml",
                        help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

def get_neighbour_dic(_cfg):
    """For a short_name image, get its nearest neighbour."""

    neighbour_dic = {}
    image_dir = os.path.join(_cfg.PATH.SEARCH_DIR,
                             _cfg.SELF_TRAIN.SOURCE)
    neighbour_lenth = _cfg.SELF_TRAIN.QUERY_SEED_NEIGHBOUR
    for device_id in os.listdir(image_dir):
        device_dir = os.path.join(image_dir, device_id)
        image_files = sorted(os.listdir(device_dir))
        image_files = [f'{device_id}/{item}' for item in image_files]
        for center_index, center_name in enumerate(image_files):
            left_index = max(0, center_index-neighbour_lenth)
            right_index = min(center_index+neighbour_lenth+1, len(image_files))
            gallery_list = image_files[left_index: right_index]
            assert len(gallery_list) > 0
            neighbour_dic[center_name] = [{'image_name': item} for item in gallery_list]
    return neighbour_dic


def generate_query_seed(_cfg):
    """Merge multiple source of query seed."""

    merged_bboxes = []
    merged_sims = []
    unique_image_ids = []
    repeat_times = []
    query_seed_type = _cfg.SELF_TRAIN.QUERY_SEED_TYPE
    shared_query_seed = f'./tests/features/{_cfg.EVAL.DATA_MODE}/'\
                        f'{query_seed_type}.pkl'
    all_pred_dic = pickle.load(open(shared_query_seed, 'rb'))
    pred_dic = list(all_pred_dic.values())[0]
    count = 0
    for image_name, pred_for_large in pred_dic.items():
        indexes = np.argsort(-pred_for_large['sim'])[:_cfg.INFERENCE.LOCALIZER_TOPK]
        if len(indexes) == 0:
            continue
        merged_bboxes.append(pred_for_large['bbox'][indexes])
        merged_sims.append(pred_for_large['sim'][indexes])
        repeat_times.append(len(pred_for_large['bbox'][indexes]))
        unique_image_ids.append(image_name)
        count += len(indexes)
    print(f'{count} query seed for {query_seed_type}')
    bboxes = np.concatenate(merged_bboxes)
    pred_sim = np.concatenate(merged_sims)
    image_ids = np.repeat(unique_image_ids, repeat_times)
    print(f'{len(pred_sim)} query seed in total')
    return {'sim': pred_sim, 'bbox': bboxes, 'image_name': image_ids}


def expand_neighbour(_cfg, query_seed, neighbour_dic):
    """Expand neighbour."""

    query_seed_name = _cfg.SELF_TRAIN.QUERY_SEED_FILE
    query_instance_dic = {}
    image_ids = query_seed['image_name']
    bboxes = query_seed['bbox']
    for index, image_name in enumerate(image_ids):
        query_instance_id = f'query_seed_{index:08d}'
        bbox = bboxes[index]
        device_id = image_name.split('/')[0]
        neighbour_gallery_list = neighbour_dic[image_name]
        bbox_dic = {'device_id': device_id,
                    'bbox': bbox.astype('int'),
                    'object_id': str(index),
                    'raw_image_name': image_name,
                    'image_name': image_name,
                    'gallery_list': neighbour_gallery_list,
                    'pos_gallery_list': neighbour_gallery_list,
                    'ignore': 0}
        query_instance_dic[query_instance_id] = bbox_dic
    query_seed_dir = os.path.dirname(query_seed_name)
    if not os.path.exists(query_seed_dir):
        os.makedirs(query_seed_dir)
    pickle.dump(query_instance_dic, open(query_seed_name, 'wb'))

def vis_query_seed(_cfg, query_seed):
    """Vis query seed."""

    vis_dir = _cfg.SELF_TRAIN.QUERY_SEED_FILE.replace('query_seed.pkl', 'vis_seed')
    query_seed_type = _cfg.SELF_TRAIN.QUERY_SEED_TYPE
    vis_dir = f'./tests/features/{_cfg.EVAL.DATA_MODE}/../vis_pred/{query_seed_type}'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    count = 0
    for image_name, bbox in zip(query_seed['image_name'], query_seed['bbox']):
        if count > 10:    # just for vis
            continue
        img = cv2.imread(os.path.join(_cfg.PATH.SEARCH_DIR,\
                        _cfg.SELF_TRAIN.SOURCE, image_name))
        print(img.shape, bbox)
        bbox = bbox.astype('int')
        # height, width = img.shape[:2]
        # bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width)
        # bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height)
        patch = img[bbox[1]: bbox[3], bbox[0]: bbox[2], :]
        cv2.imwrite(os.path.join(vis_dir, os.path.basename(image_name)), patch)
        count += 1


def main():
    """Main function."""

    args = parse_args()
    if args.config_file != "":
        cfg.merge_from_file(os.path.join('./instance_search/config/',
                                         args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    neighbour_dic = get_neighbour_dic(cfg)
    query_seed = generate_query_seed(cfg)
    vis_query_seed(cfg, query_seed)
    expand_neighbour(cfg, query_seed, neighbour_dic)


if __name__ == "__main__":
    main()
