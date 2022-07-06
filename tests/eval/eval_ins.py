"""Eval INS.
Author: gongyou.zyq
Date: 2020.11.25
"""

import os
import pickle
import time

import numpy as np

from instance_search.config import cfg
from instance_search.utils.io_utils import parse_args
from tests.eval.base_evaluator import BaseEvaluator


def compute_iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """

    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    # pylint: disable=invalid-name
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    # pylint: enable=invalid-name
    return o


class INSEvaluator(BaseEvaluator):
    """INS evaluator."""

    def __init__(self, _cfg):
        BaseEvaluator.__init__(self, _cfg)

    def check_bbox_match(self, bbox_pred, bbox_gt):
        """Check whether gt_bbox and pred_bbox overlap."""

        bbox_gt = bbox_gt.astype(float)
        ovmax = compute_iou(bbox_gt, bbox_pred)
        match_flag = ovmax > self._cfg.EVAL.IOU_THR
        return match_flag

    def get_matching_flag(self, gt_info, pred_info):
        """Get matching flag"""

        image_ids = pred_info['image_ids']
        sorted_sim = pred_info['sorted_sim']
        pred_bboxes = pred_info['pred_bboxes']
        ignore_list = gt_info['ignore_list']
        gt_bbox_dic = gt_info['gt_bbox_dic']
        gt_matched_flag = gt_info['gt_matched_flag']

        valid_flag = np.in1d(image_ids, ignore_list, invert=True)
        # select valid data
        image_ids = image_ids[valid_flag]
        pred_bboxes = pred_bboxes[valid_flag]
        sorted_sim = sorted_sim[valid_flag]
        # 0 0 1 0 0
        gt_contain_flag = np.in1d(image_ids, list(gt_bbox_dic.keys()))
        tp_flag = np.zeros(len(image_ids))
        for rank in np.argwhere(gt_contain_flag > 0).flatten():
            image_name = image_ids[rank]
            if self.check_bbox_match(pred_bboxes[rank, :],
                                     gt_bbox_dic[image_name]):
                if not gt_matched_flag[image_name]:
                    tp_flag[rank] = 1.
                    gt_matched_flag[image_name] = 1
        fp_flag = 1.0 - tp_flag
        return tp_flag, fp_flag, sorted_sim, gt_matched_flag, valid_flag


def main():
    """Main method"""

    args = parse_args()
    if args.config_file != "":
        cfg.merge_from_file(os.path.join('./instance_search/config/',
                                         args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    start_time = time.time()
    feature_dir = f'./tests/features/{cfg.EVAL.DATA_MODE}'
    all_loc_info = pickle.load(open(
        f'{feature_dir}/{cfg.EVAL.ROUGH_LOCALIZER}.pkl', 'rb'))
    all_reid_info = pickle.load(open(
        f'{feature_dir}/{cfg.EVAL.SIM_MODE}.pkl', 'rb'))
    print(f'{time.time() - start_time} seconds to load pred')

    ins_evaluator = INSEvaluator(cfg)
    ins_evaluator.eval_data(all_loc_info, all_reid_info)
    ins_evaluator.output_final_result()


if __name__ == "__main__":
    main()
