"""Test localizer with a pair of images.
Author: gongyou.zyq
Date: 2020.11.18
"""

import os
import pickle
import time

import cv2
import numpy as np

from instance_search.instance_search import localizer_factory
from instance_search.utils.io_utils import get_query_info, get_gallery_info,\
                                           parse_args, select_good_id
from instance_search.config import cfg

GALLERY_SIZE = 100
QUERY_SIZE = 5
GALLERY_BATCH_SIZE = 5    # not very sensitive to gallery batch size
# QUERY_SIZE = 2500
# GALLERY_BATCH_SIZE = 1


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


def debug_one_query_one_gallery():
    """Debug one query + one image."""

    args = parse_args()
    if args.config_file != "":
        cfg.merge_from_file(os.path.join('./instance_search/config/',
                                         args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    query_large_path, query_bbox, gallery_large_path = load_sample_data(cfg)
    localizer = localizer_factory(0, cfg)
    query_img_bgr = cv2.imread(query_large_path)
    localizer.set_query(query_img_bgr, query_bbox, 'query_instance_id')
    activated_query_list = ['query_instance_id']
    localizer.set_activated_query(activated_query_list)
    gallery_img_bgr = cv2.imread(gallery_large_path)
    for _ in range(10):
        det_result_queries = localizer.detect([gallery_img_bgr])
    start_time = time.time()
    localizer.reset()
    for i in range(QUERY_SIZE):
        localizer.set_query(query_img_bgr, query_bbox, f'query_{i}')
        localizer.set_activated_query([f'query_{i}'])
        for _ in range(GALLERY_SIZE):
            det_result_queries = localizer.detect([gallery_img_bgr])
        localizer.reset()
    duration = (time.time()-start_time)*1000.0/(GALLERY_SIZE * QUERY_SIZE)
    print(f'time for {cfg.EVAL.ROUGH_LOCALIZER} one image: {duration:.4f} ms,'
          f'fps: {int(1000.0/duration)}')
    # We know this is only one query
    pred_bboxes = det_result_queries[f'query_{QUERY_SIZE-1}'][0]
    localizer.draw([query_large_path, query_bbox],
                   [gallery_large_path, pred_bboxes],
                   f'./tests/images/debug_localizer/{cfg.EVAL.DATASET_NAME}/'
                   f'debug_{cfg.EVAL.ROUGH_LOCALIZER}.jpg')


def debug_multi_query_multi_gallery():
    """Debug multi query + multi gallery image."""

    args = parse_args()
    if args.config_file != "":
        cfg.merge_from_file(os.path.join('./instance_search/config/',
                                         args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    query_large_path, query_bbox, gallery_large_path = load_sample_data(cfg)
    localizer = localizer_factory(0, cfg)
    query_img_bgr = cv2.imread(query_large_path)
    activated_query_list = []
    for i in range(QUERY_SIZE):
        localizer.set_query(query_img_bgr, query_bbox, f'query_{i}')
        activated_query_list.append(f'query_{i}')
    gallery_img_bgr = cv2.imread(gallery_large_path)
    localizer.set_activated_query(activated_query_list)
    for _ in range(10):
        det_result_queries = localizer.detect(
                [gallery_img_bgr] * GALLERY_BATCH_SIZE)
    start_time = time.time()
    localizer.reset()
    for i in range(QUERY_SIZE):
        localizer.set_query(query_img_bgr, query_bbox, f'query_{i}')
    localizer.set_activated_query(activated_query_list)
    num_batches = int(GALLERY_SIZE/GALLERY_BATCH_SIZE)
    for _ in range(num_batches):
        det_result_queries = localizer.detect(
                [gallery_img_bgr] * GALLERY_BATCH_SIZE)
    localizer.reset()
    duration = (time.time()-start_time)*1000.0/(GALLERY_SIZE * QUERY_SIZE)
    print(f'time for {cfg.EVAL.ROUGH_LOCALIZER} one image: {duration:.4f} ms,'
          f'fps: {int(1000.0/duration)}')
    # We know this is only one query
    pred_bboxes = det_result_queries[f'query_{QUERY_SIZE-1}'][0]
    localizer.draw([query_large_path, query_bbox],
                   [gallery_large_path, pred_bboxes],
                   f'./tests/images/debug_localizer/{cfg.EVAL.DATASET_NAME}/'
                   f'debug_{cfg.EVAL.ROUGH_LOCALIZER}.jpg')


def debug_seq():
    """Debug seq."""

    args = parse_args()
    if args.config_file != "":
        cfg.merge_from_file(os.path.join('./instance_search/config/',
                                         args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    seq_dir = '/mnt1/gongyou.zyq/data/OTB'
    seq_name = 'Basketball'
    image_dir = f'{seq_dir}/{seq_name}/img/'
    anno_file = f'{seq_dir}/{seq_name}/groundtruth_rect.txt'
    with open(anno_file, 'r') as f_txt:
        bbox = f_txt.readlines()[0].strip().split(',')
    bbox = np.array(bbox).astype('int')
    bbox[2:] += bbox[:2]

    localizer = localizer_factory(0, cfg)
    img_bgr = cv2.imread(image_dir + '0001.jpg')
    localizer.set_query(img_bgr, bbox, 'query_0')
    localizer.set_activated_query(['query_0'])
    debug_seq_dir = f'tests/images/debug_seq/{cfg.EVAL.ROUGH_LOCALIZER}'
    if not os.path.exists(debug_seq_dir):
        os.makedirs(debug_seq_dir)
    for item in sorted(os.listdir(image_dir))[1:]:
        img_bgr = cv2.imread(os.path.join(image_dir, item))
        det_result = localizer.detect([img_bgr])
        det_result = det_result['query_0'][0]
        orders = np.argsort(-det_result[:, -1])
        for index in orders[:1]:
            out_bbox = det_result[index][:4]
            print(item, out_bbox)
            out_bbox = out_bbox.astype('int')
            cv2.rectangle(img_bgr, (out_bbox[0], out_bbox[1]),
                          (out_bbox[2], out_bbox[3]), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_seq_dir, item), img_bgr)
    localizer.reset()


def main():
    """Main function."""

    # debug_one_query_one_gallery()
    debug_multi_query_multi_gallery()
    # debug_seq()


if __name__ == "__main__":
    main()
