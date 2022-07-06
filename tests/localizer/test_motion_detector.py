"""Test localizer with a pair of images.
Author: gongyou.zyq
Date: 2020.11.18
"""

from glob import glob
import os
import pickle
import time

import cv2

from instance_search.motion_detection import motion_detection_factory
from instance_search.utils.io_utils import get_query_info, get_gallery_info,\
                                           parse_args, select_good_id,\
                                           save_pickle
from instance_search.config import cfg


def load_sample_data(_cfg):
    """Select a query_instance_id and the corresponding gallery data."""

    query_instance_dic = pickle.load(open(_cfg.EVAL.LABEL_FILE, 'rb'))
    selected_test_id = select_good_id(_cfg, query_instance_dic)
    query_instance_id = selected_test_id[0]
    query_bbox_dic = query_instance_dic[query_instance_id]
    query_large_path, query_bbox = get_query_info(_cfg, query_bbox_dic)
    object_id = query_bbox_dic['object_id']
    gallery_large_path, gallery_bbox = get_gallery_info(
            _cfg, query_bbox_dic['pos_gallery_list'], sample_index=0)
    print(f'Query instance id: {query_instance_id}, object id: {object_id}, '
          f'query_large_path: {query_large_path},  bbox: {query_bbox}')
    print(f'Gallery large path: {gallery_large_path}, bbox: {gallery_bbox}')
    return query_large_path, query_bbox, gallery_large_path


def debug_seq():    # pylint: disable=too-many-locals
    """Debug seq."""

    args = parse_args()
    if args.config_file != "":
        cfg.merge_from_file(os.path.join('./instance_search/config/',
                                         args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    query_large_path, query_bbox, gallery_large_path = load_sample_data(cfg)
    motion_detector = motion_detection_factory(0, cfg)
    image_dir = cfg.PATH.SEARCH_DIR
    start_time = time.time()
    count = 0
    result_dic = {}
    for device_id in os.listdir(image_dir):
        device_dir = os.path.join(image_dir, device_id)
        print(device_id, count)
        image_list = glob(device_dir + '/*.jpg')
        image_list = sorted(image_list)
        for image_file in image_list:
            img_bgr = cv2.imread(image_file)
            det_result = motion_detector.detect(img_bgr)
            gallery_large_path = image_file
            count += 1
            image_name = os.path.basename(image_file)
            short_name = f'{device_id}/{image_name}'
            result_dic[short_name] = det_result
        # very important
        motion_detector.reset()
    duration = (time.time()-start_time)*1000.0/count
    print(f'time for {cfg.EVAL.ROUGH_LOCALIZER} one image: {duration:.4f} ms,'
          f'fps: {int(1000.0/duration)}')
    save_dir = f'./tests/images/debug_localizer/{cfg.EVAL.DATASET_NAME}'
    motion_detector.draw([query_large_path, query_bbox],
                         [gallery_large_path, det_result],
                         f'{save_dir}/debug_{cfg.EVAL.ROUGH_LOCALIZER}.jpg')

    save_pickle(result_dic, cfg.EVAL.ROUGH_LOCALIZER, cfg.EVAL.DATA_MODE)


def main():
    """Main function."""

    debug_seq()


if __name__ == "__main__":
    main()
