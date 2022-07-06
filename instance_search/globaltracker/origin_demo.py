"""Run origin globaltrack demo.
Learn from GlobalTrack/tools/test_global_track.py.
"""
# pylint: disable=line-too-long, wrong-import-position, unused-import

import sys
import pickle
import cv2

sys.path.append('./instance_search/globaltracker/GlobalTrack/')
sys.path.append('./instance_search/globaltracker/GlobalTrack/_submodules/neuron')
sys.path.append('./instance_search/globaltracker/GlobalTrack/_submodules/mmdetection')
import _init_paths
import neuron.data as data
from trackers import GlobalTrack

from instance_search.utils.io_utils import get_object_info, select_good_id
from instance_search.config import cfg

def run_origin_globaltrack():
    """Run origin globaltrack demo."""

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

    query_img_bgr = cv2.imread(query_large_path)
    gallery_img_bgr = cv2.imread(gallery_large_path)
    cfg_file = './instance_search/globaltracker/GlobalTrack/configs/qg_rcnn_r50_fpn.py'
    ckp_file = './instance_search/globaltracker/GlobalTrack/log/baseline/qg_rcnn_r50_fpn_coco_got10k_lasot.pth'
    transforms = data.BasicPairTransforms(train=False)
    tracker = GlobalTrack(
        cfg_file, ckp_file, transforms,
        name_suffix='qg_rcnn_r50_fpn')
    tracker.init(query_img_bgr, query_bbox)
    det_result = tracker.update(gallery_img_bgr)
    print('gallery path: ', gallery_large_path)
    print('pred: ', det_result)
    print('label: ', gallery_bbox)

if __name__ == "__main__":
    run_origin_globaltrack()
