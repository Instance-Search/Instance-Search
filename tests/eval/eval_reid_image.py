# -*- coding:utf-8 -*-
"""
Load precomputed feature.pkl and compute CMC and mAP for reid.
Author: Yuqi Zhang
Date: 2019.08.29
"""

import os

import numpy as np

from instance_search.config import cfg
from instance_search.utils.io_utils import parse_args
import tests.eval.cmc as cmc
from tests.eval.extract_cropped_reid import extract_feature


class ReIDAccuracy:
    """ReID evaluation."""

    def __init__(self, _cfg):

        self.test_probe, self.test_gallery = extract_feature(cfg)

    @staticmethod
    def get_data(feature_dic):
        """Get test set."""

        test_feature = []
        test_label = []
        set_id = set()
        for file_name in feature_dic:
            short_name = file_name.split('.')[0]
            try:
                rawid, cam, _ = short_name.split('_')
                set_id.add(rawid)
                rawid = int(rawid)
                cam = int(cam)
            except ValueError:    # distractor
                rawid = -1
                cam = -1
            test_feature.append(feature_dic[file_name])
            test_label.append([rawid, cam])
        print(f'{len(set_id)} unique ids')
        return np.array(test_feature), np.array(test_label)

    def get_result(self):
        """Compute reid rank-1 and mAP."""

        gal_feats, gal_labels = self.get_data(self.test_gallery)
        prb_feats, prb_labels = self.get_data(self.test_probe)

        sims = cmc.ComputeEuclid(prb_feats, gal_feats, fg_sqrt=True,
                                 fg_norm=True)
        r_cmc = cmc.GetRanks(sims, prb_labels, gal_labels, 10, True)
        r_map = cmc.GetMAP(sims, prb_labels, gal_labels, True)
        print('Model: Separate Camera VCS cmc=%s, map=%s', r_cmc, r_map)

        sims = cmc.ComputeEuclid(prb_feats, gal_feats, fg_sqrt=True,
                                 fg_norm=True)
        r_cmc = cmc.GetRanks(sims, prb_labels, gal_labels, 10, False)
        r_map = cmc.GetMAP(sims, prb_labels, gal_labels, False)
        print('Model: No Separate Camera VCS cmc=%s, map=%s', r_cmc, r_map)


def main():
    """Main function."""

    args = parse_args()
    if args.config_file != "":
        cfg.merge_from_file(os.path.join('./instance_search/config/',
                                         args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    reid_acc = ReIDAccuracy(cfg)
    reid_acc.get_result()


if __name__ == "__main__":
    main()
