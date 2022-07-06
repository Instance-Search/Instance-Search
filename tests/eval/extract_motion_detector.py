"""Test localizer with a pair of images.
Author: gongyou.zyq
Date: 2020.11.18
"""

import pickle
import time
import shutil
from glob import glob
import os
import argparse
import cv2

from instance_search.motion_detection import motion_detection_factory
from instance_search.utils.io_utils import select_good_id, save_pickle
from instance_search.config import cfg

def parse_args():
    """Parse args."""

    parser = argparse.ArgumentParser(description="Instance Search Config")
    parser.add_argument("--config_file", default="Cityflow.yml",
                        help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def extract_seq(_cfg):
    """Extract seq."""

    motion_detector = motion_detection_factory(0, _cfg)
    image_dir = os.path.join(_cfg.PATH.SEARCH_DIR, _cfg.EVAL.GALLERY_DATA_SOURCE)
    start_time = time.time()
    count = 0
    result_dic = {}
    for device_id in os.listdir(image_dir):
        device_dir = os.path.join(image_dir, device_id)
        print(device_id, count)
        for image_file in sorted(glob(device_dir + '/*.jpg')):
            img_bgr = cv2.imread(image_file)
            det_result = motion_detector.detect(img_bgr)
            count += 1
            image_name = os.path.basename(image_file)
            short_name = f'{device_id}/{image_name}'
            result_dic[short_name] = det_result
        motion_detector.reset()
    duration = (time.time()-start_time)*1000.0/count
    print(f'time for {_cfg.EVAL.ROUGH_LOCALIZER} one image: {duration:.4f} ms,'\
          f'fps: {int(1000.0/duration)}')
    save_pickle({'shared': result_dic},
                _cfg.EVAL.ROUGH_LOCALIZER,
                _cfg.EVAL.DATA_MODE)

def copy_pred(_cfg):
    """Copy pred files for each query instance id, for eval purpose only."""

    query_instance_dic = pickle.load(open(_cfg.EVAL.LABEL_FILE, 'rb'))
    selected_test_id = select_good_id(_cfg, query_instance_dic)
    feature_dir = (f'./tests/features/{_cfg.EVAL.DATA_MODE}/'
                   f'{_cfg.EVAL.ROUGH_LOCALIZER}/')
    for query_instance_id in selected_test_id[:cfg.EVAL.TEST_QUERY_NUM]:
        shutil.copy(os.path.join(feature_dir, 'shared.pkl'),
                    os.path.join(feature_dir, f'{query_instance_id}.pkl'))

def main():
    """Main function."""

    args = parse_args()
    if args.config_file != "":
        cfg.merge_from_file(os.path.join('./instance_search/config/',
                                         args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    extract_seq(cfg)
    # copy_pred(cfg)


if __name__ == "__main__":
    main()
