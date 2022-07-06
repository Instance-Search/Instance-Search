"""Run origin siamrpn demo.
Learn from pysot/tools.
"""
# pylint: disable=too-many-locals, line-too-long, wrong-import-position

import sys
import pickle
import cv2
import numpy as np
import torch

sys.path.append('./instance_search/siamrpn_tracker/pysot/')
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
import pysot.core.config as siamese_cfg

from instance_search.utils.io_utils import get_object_info, select_good_id
from instance_search.config import cfg

def run_origin_siamrpn():
    """Run origin siamrpn demo."""

    cfg.merge_from_file('./instance_search/config/Cityflow.yml')
    cfg.freeze()

    query_dic, gallery_dic = pickle.load(open(cfg.EVAL.LABEL_FILE, 'rb'))
    object_id = select_good_id(cfg, gallery_dic)[4]
    query_large_path, query_bbox = get_object_info(cfg, query_dic[object_id],
                                                   sample_index=0)
    gallery_large_path, gallery_bbox = get_object_info(cfg, gallery_dic[object_id],
                                                       sample_index=1)
    print(f'Query object_id: {object_id}, large_path: {query_large_path}, '
          f' bbox: {query_bbox}')

    tracker_cfg = siamese_cfg.cfg
    tracker_cfg.merge_from_file('./instance_search/siamrpn_tracker/pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml')
    tracker_cfg.CUDA = torch.cuda.is_available() and tracker_cfg.CUDA
    device = torch.device("cuda" if tracker_cfg.CUDA else "cpu")
    model = ModelBuilder()
    # load model
    snapshot = './instance_search/siamrpn_tracker/pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/model.pth'
    model.load_state_dict(
        torch.load(snapshot, map_location=lambda storage, loc: storage.cpu())
    )
    model.eval().to(device)
    # build tracker
    tracker = build_tracker(model)

    query_img_bgr = cv2.imread(query_large_path)
    siamese_bbox = np.array([query_bbox[0], query_bbox[1],\
                             query_bbox[2]-query_bbox[0],\
                             query_bbox[3]-query_bbox[1]])
    tracker.init(query_img_bgr, siamese_bbox)
    gallery_img_bgr = cv2.imread(gallery_large_path)
    det_result = tracker.track(gallery_img_bgr)
    print('gallery path: ', gallery_large_path)
    print('pred: ', det_result)
    print('label: ', gallery_bbox)

if __name__ == "__main__":
    run_origin_siamrpn()
