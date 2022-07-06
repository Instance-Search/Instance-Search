"""Test refiner. localizer results needed first
Author: gongyou.zyq
Date: 2020.11.19
"""

import time
import argparse
import os
import numpy as np
import cv2

from instance_search.instance_search import refine_factory
from instance_search.config import cfg

def parse_args():
    """Parse args."""

    parser = argparse.ArgumentParser(description="Instance Search Config")
    parser.add_argument("--config_file", default="",
                        help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

def test_small(reid, img):
    """Test small image without batch mode"""

    start_time = time.time()
    num_loops = 10000
    for _ in range(num_loops):
        feat = reid.extract(img)
    duration = (time.time()-start_time)*1000.0/num_loops
    fps = 1000.0 / duration
    print('Test small image without batch mode: %.4f ms, fps: %d' % (duration, int(fps)))
    print(f'feature shape: {feat.shape}')

def test_small_batch(reid, img):
    """Test small image with batch mode."""

    num_loops = 10000
    img_list = [img]*num_loops
    start_time = time.time()
    feat = reid.extract_batch(img_list)
    duration = time.time() - start_time
    duration = (time.time()-start_time)*1000.0/num_loops
    fps = 1000.0 / duration
    print('Test small image with batch mode: %.4f ms, fps: %d' % (duration, int(fps)))
    print(f'feature shape: {feat.shape}')

def test_bboxes(reid, large_img, bboxes):
    """Test bboxes without batch mode."""

    num_loops = 1000
    start_time = time.time()
    for _ in range(num_loops):
        feat = reid.extract_bboxes_nobatch(large_img, bboxes)
    duration = time.time() - start_time
    duration = (time.time()-start_time)*1000.0/num_loops
    fps = 1000.0 / duration
    print('Test bboxes without batch mode: %.4f ms, fps: %d' % (duration, int(fps)))
    print(f'feature shape: {feat.shape}')

def test_bboxes_batch(reid, large_img, bboxes):
    """Test bboxes in batch mode."""

    num_loops = 1000
    start_time = time.time()
    for _ in range(num_loops):
        feat = reid.extract_bboxes_batch(large_img, bboxes)
    duration = time.time() - start_time
    duration = (time.time()-start_time)*1000.0/num_loops
    fps = 1000.0 / duration
    print('Test bboxes in batch mode: %.4f ms, fps: %d' % (duration, int(fps)))
    print(f'feature shape: {feat.shape}')

def main():
    """Main."""

    args = parse_args()
    if args.config_file != "":
        cfg.merge_from_file(os.path.join('./instance_search/config/',
                                         args.config_file))
    cfg.freeze()

    gpus = '0'
    reid = refine_factory(gpus, cfg)
    imgpath = '/home/gongyou.zyq/datasets/Corrected_Market1501/trainval/0448_04_0001.jpg'
    img = cv2.imread(imgpath)
    # img = np.zeros((100, 100, 3), dtype=np.uint8)
    large_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bboxes = np.array([[0, 0, 100, 100]]*16)
    # warmup
    for _ in range(10):
        reid.extract(img)

    test_small(reid, img)
    test_small_batch(reid, img)
    test_bboxes(reid, large_img, bboxes)
    test_bboxes_batch(reid, large_img, bboxes)


if __name__ == '__main__':
    main()
