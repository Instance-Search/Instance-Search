"""Eval CBIR.
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


class CBIREvaluator(BaseEvaluator):
    """CBIR Evaluator."""

    def __init__(self, _cfg):
        BaseEvaluator.__init__(self, _cfg)

    def get_matching_flag(self, gt_info, pred_info):
        """Get matching flag"""

        image_ids = pred_info['image_ids']
        sorted_sim = pred_info['sorted_sim']
        ignore_list = gt_info['ignore_list']
        gt_bbox_dic = gt_info['gt_bbox_dic']
        gt_matched_flag = gt_info['gt_matched_flag']

        gt_name_list = list(gt_bbox_dic.keys())
        valid_flag = np.zeros(len(image_ids))
        tp_flag = valid_flag.copy()
        fp_flag = valid_flag.copy()
        thrs = valid_flag.copy()
        for rank, image_name in enumerate(image_ids):
            if image_name not in ignore_list:
                valid_flag[rank] = 1
            thrs[rank] = sorted_sim[rank]
            if image_name in gt_name_list:
                tp_flag[rank] = 1.
                gt_matched_flag[image_name] = 1
            else:
                fp_flag[rank] = 1.
        tp_flag = tp_flag[valid_flag > 0]
        fp_flag = fp_flag[valid_flag > 0]
        thrs = thrs[valid_flag > 0]
        valid_flag = np.where(valid_flag>0)[0]
        return tp_flag, fp_flag, thrs, gt_matched_flag, valid_flag


def main():
    """Main method"""

    args = parse_args()
    if args.config_file != "":
        cfg.merge_from_file(os.path.join('./instance_search/config/',
                                         args.config_file))
    cfg.merge_from_list(args.opts)
    # leave only one most confident bbox per large image
    cfg.EVAL.TOPK_PER_LARGE = 1
    cfg.freeze()

    start_time = time.time()
    feature_dir = f'./tests/features/{cfg.EVAL.DATA_MODE}'
    all_loc_info = pickle.load(open(
        f'{feature_dir}/{cfg.EVAL.ROUGH_LOCALIZER}.pkl', 'rb'))
    all_reid_info = pickle.load(open(
        f'{feature_dir}/{cfg.EVAL.SIM_MODE}.pkl', 'rb'))
    print(f'{time.time() - start_time} seconds to load pred')

    cbir_evaluator = CBIREvaluator(cfg)
    cbir_evaluator.eval_data(all_loc_info, all_reid_info)
    cbir_evaluator.output_final_result()


if __name__ == "__main__":
    main()
