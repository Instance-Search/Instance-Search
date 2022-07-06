"""Test instance searcher with a pair of images.
This may have overlap with test_localizer

Author: gongyou.zyq
Date: 2020.11.28
"""
import os
import pickle
import time

import cv2

from instance_search.instance_search import InstanceSearcher
from instance_search.utils.io_utils import get_query_info, get_gallery_info,\
                                           parse_args, select_good_id
from instance_search.config import cfg

GALLERY_SIZE = 20
QUERY_SIZE = 20
GALLERY_BATCH_SIZE = 5
# QUERY_SIZE = 2500
# GALLERY_BATCH_SIZE = 1


def load_sample_data(_cfg):
    """Select a query_instance_id and the corresponding gallery data."""

    query_instance_dic = pickle.load(open(_cfg.EVAL.LABEL_FILE, 'rb'))
    selected_test_id = select_good_id(_cfg, query_instance_dic)
    query_instance_id = selected_test_id[0]
    query_bbox_dic = query_instance_dic[query_instance_id]
    query_large_path, query_bbox = get_query_info(_cfg, query_bbox_dic)
    query_bbox = query_bbox.astype('int')
    object_id = query_bbox_dic['object_id']
    gallery_large_path, gallery_bbox = get_gallery_info(
            _cfg, query_bbox_dic['pos_gallery_list'], sample_index=0)
    print(f'Query instance id: {query_instance_id}, object id: {object_id}, '
          f'query_large_path: {query_large_path},  bbox: {query_bbox}')
    print(f'Gallery large path: {gallery_large_path}, bbox: {gallery_bbox}')
    return query_large_path, query_bbox, gallery_large_path


def debug_one_query_one_gallery():
    """Debug instance search."""

    args = parse_args()
    if args.config_file != "":
        cfg.merge_from_file(os.path.join('./instance_search/config/',
                                         args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    query_large_path, query_bbox, gallery_large_path = load_sample_data(cfg)
    instance_searcher = InstanceSearcher(0, cfg)
    query_img_bgr = cv2.imread(query_large_path)
    instance_searcher.set_query(query_img_bgr, query_bbox, 'query_instance_id')
    activated_query_list = ['query_instance_id']
    instance_searcher.set_activated_query(activated_query_list)
    gallery_img_bgr = cv2.imread(gallery_large_path)
    for _ in range(10):
        _ = instance_searcher.detect([gallery_img_bgr])
    start_time = time.time()
    instance_searcher.reset()
    for i in range(QUERY_SIZE):
        instance_searcher.set_query(query_img_bgr, query_bbox, f'query_{i}')
        instance_searcher.set_activated_query([f'query_{i}'])
        for _ in range(GALLERY_SIZE):
            rough_det_queries, refine_det_queries = \
                    instance_searcher.detect([gallery_img_bgr])
    duration = (time.time()-start_time)*1000.0/(GALLERY_SIZE * QUERY_SIZE)
    print(f'time for {cfg.EVAL.ROUGH_LOCALIZER} one image: {duration:.4f} ms,'
          f'fps: {int(1000.0/duration)}')
    # We know this is only one query
    pred_bboxes = rough_det_queries[f'query_{QUERY_SIZE-1}'][0]
    instance_searcher.draw([query_large_path, query_bbox],
                           [gallery_large_path, pred_bboxes],
                           './tests/images/debug_rough.jpg')
    pred_bboxes = refine_det_queries[f'query_{QUERY_SIZE-1}'][0]
    instance_searcher.draw([query_large_path, query_bbox],
                           [gallery_large_path, pred_bboxes],
                           './tests/images/debug_refine.jpg')


def debug_multi_query_multi_gallery():    # pylint: disable=too-many-locals
    """Debug instance search."""

    args = parse_args()
    if args.config_file != "":
        cfg.merge_from_file(os.path.join('./instance_search/config/',
                                         args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    query_large_path, query_bbox, gallery_large_path = load_sample_data(cfg)
    instance_searcher = InstanceSearcher(0, cfg)
    query_img_bgr = cv2.imread(query_large_path)
    activated_query_list = []
    for i in range(QUERY_SIZE):
        instance_searcher.set_query(query_img_bgr, query_bbox, f'query_{i}')
        activated_query_list.append(f'query_{i}')
    gallery_img_bgr = cv2.imread(gallery_large_path)
    instance_searcher.set_activated_query(activated_query_list)
    for _ in range(1):
        rough_det_queries, refine_det_queries = instance_searcher.detect(
                [gallery_img_bgr]*GALLERY_BATCH_SIZE)
    start_time = time.time()
    instance_searcher.reset()
    for i in range(QUERY_SIZE):
        instance_searcher.set_query(query_img_bgr, query_bbox, f'query_{i}')
    instance_searcher.set_activated_query(activated_query_list)
    num_batches = int(GALLERY_SIZE/GALLERY_BATCH_SIZE)
    for _ in range(num_batches):
        rough_det_queries, refine_det_queries = instance_searcher.detect(
                [gallery_img_bgr]*GALLERY_BATCH_SIZE)
        print(refine_det_queries[f'query_{QUERY_SIZE-1}'])
    duration = (time.time()-start_time)*1000.0/(GALLERY_SIZE * QUERY_SIZE)
    print(f'time for {cfg.EVAL.ROUGH_LOCALIZER} one image: {duration:.4f} ms,'
          f'fps: {int(1000.0/duration)}')
    # We know this is only one query
    pred_bboxes = rough_det_queries[f'query_{QUERY_SIZE-1}'][0]
    instance_searcher.draw([query_large_path, query_bbox],
                           [gallery_large_path, pred_bboxes],
                           './tests/images/debug_rough.jpg')
    pred_bboxes = refine_det_queries[f'query_{QUERY_SIZE-1}'][0]
    instance_searcher.draw([query_large_path, query_bbox],
                           [gallery_large_path, pred_bboxes],
                           './tests/images/debug_refine.jpg')


if __name__ == "__main__":
    debug_one_query_one_gallery()
    # debug_multi_query_multi_gallery()
