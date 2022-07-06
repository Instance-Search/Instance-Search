"""Extract delg features and global-local matching.
Author: gongyou.zyq
Date: 2020.11.26
"""

import os
import pickle
import sys

import cv2
import numpy as np

from instance_search.base_matcher import BaseMatcher
sys.path.append('/home/gongyou.zyq/TransMatching/')
from config import cfg
from inference import VitMatcher

DEBUG = False


class DelgVitMatcher(BaseMatcher):
    """Delg global local inference."""

    def __init__(self, gpus, _cfg):

        BaseMatcher.__init__(self, gpus, _cfg)
        cfg.merge_from_file('/home/gongyou.zyq/TransMatching/configs/'
                            'GLDv2/pairDeiTsmall_256.yml')
        # cfg.TEST.WEIGHT = ('/home/gongyou.zyq/TransMatching/logs/'
        #                    'GLDv2clean/pairViT_256/transformer_5.pth')

        # cfg.merge_from_file('/home/gongyou.zyq/TransMatching/configs/'
        #                     'GLDv2/R50_384.yml')
        # cfg.MODEL.NAME = 'resnet101_ibn_a'
        # cfg.INPUT.SIZE_TEST = [384, 384]
        # cfg.MODEL.PRETRAIN_PATH = '/home/gongyou.zyq/.torch/r50_ibn_a.pth'
        # cfg.TEST.WEIGHT = model_path

        self.vit_matcher = VitMatcher(cfg)

        # NOTE: For debug, not used by Vit local matching code
        if DEBUG:
            feat_dir = './tests/features/'
            debug_query = 'Oxford5k/baseline/delg/query_cropresize.pkl'
            debug_query = os.path.join(feat_dir, debug_query)
            debug_index = 'Oxford5k/baseline/delg/index.pkl'
            debug_index = os.path.join(feat_dir, debug_index)
            self.all_query_res = pickle.load(open(debug_query, 'rb'))
            self.all_index_res = pickle.load(open(debug_index, 'rb'))

    @staticmethod
    def extract(large_path, bbox):
        """We share the extracted feature from DELG"""

        _, _ = large_path, bbox
        # print(np.max(local_descriptor['locations'][:, 1]))
        return [], [], []

    def local_matching(self, query_res, index_name):
        """DELG local matching. See base class"""

        query_path = os.path.join(self._cfg.PATH.SEARCH_DIR, 'test_probe',
                                  query_res['image_name'])
        gallery_path = os.path.join(self._cfg.PATH.SEARCH_DIR, 'test_gallery',
                                    index_name)
        score = self.vit_matcher.match(query_path, [gallery_path])[0]
        det = np.array([[0, 0, 10000, 10000, score]])
        return det

    # pylint: disable=too-many-locals
    def local_matching_batch(self, query_res, index_name_list):
        """DELG local matching. See base class"""

        query_path = os.path.join(self._cfg.PATH.SEARCH_DIR, 'test_probe',
                                  query_res['image_name'])
        query_img = cv2.imread(query_path)
        if query_res['bbox'] is not None:
            # pylint: disable=invalid-name
            [x1, y1, x2, y2] = query_res['bbox']
            query_img = query_img[y1:y2, x1:x2, :]

        gallery_img_list = []
        for index_name in index_name_list:
            gallery_path = os.path.join(self._cfg.PATH.SEARCH_DIR,
                                        'test_gallery', index_name)
            gallery_img = cv2.imread(gallery_path)
            gallery_img_list.append(gallery_img)
        scores = self.vit_matcher.match(query_img, gallery_img_list)
        print(scores.min(), scores.max())
        local_rank_dict = {}
        for gal_index, score in enumerate(scores):
            # offset 100 to make sure local score larger than global
            det = np.array([[0, 0, 10000, 10000, score+100]])
            local_rank_dict[index_name_list[gal_index]] = det
        return local_rank_dict

    def local_matching_batch_simple(self, query_res, index_name_list):
        """DELG local matching. See base class"""

        query_feat = self.all_query_res[query_res['image_name']]['global_feat']
        gallery_feat_list = []
        for index_name in index_name_list:
            gallery_feat = self.all_index_res[index_name]['global_feat']
            gallery_feat_list.append(gallery_feat)
        gallery_feat_list = np.array(gallery_feat_list)
        scores = np.dot(gallery_feat_list, query_feat[:, np.newaxis]).flatten()
        local_rank_dict = {}
        for gal_index, score in enumerate(scores):
            # offset 100 to make sure local score larger than global
            det = np.array([[0, 0, 10000, 10000, score+100]])
            local_rank_dict[index_name_list[gal_index]] = det
        return local_rank_dict
