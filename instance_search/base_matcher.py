"""
Base keypoint extraction and global-local matching.
Author: gongyou.zyq
Date: 2020.11.26
Ref: https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
"""

from collections import OrderedDict
from multiprocessing import Pool
import pickle

import numpy as np

from instance_search.base_localizer import BaseLocalizer


# pylint: disable=fixme
class BaseMatcher(BaseLocalizer):
    """Base keypoint extraction and global-Local matcher."""

    def __init__(self, gpus, _cfg):
        BaseLocalizer.__init__(self, gpus, _cfg)
        self.query_res = None
        self.ignore_data = None
        # TODO, comment for zhexian
        self.ignore_dic = self._get_ignore_dic()
        self.rerank_num = None

    def set_query_feat(self, query_res, query_instance_id):
        """Sets query feats and info.

        Args:
            query_res: A dict of 3 element:
                        [global_feat, local_feat, bbox_info]
        """

        self.query_res = query_res
        # TODO, comment for zhexian
        self.ignore_data = self.ignore_dic[query_instance_id]

    def extract(self, large_path, bbox):
        """Extracts global local feature for one image.

        Args:
            large_path: A string for image location.
            bbox: A numpy bbox of shape (4,) dtype float32

        Returns:
            A dict with three 3 tuples:
                global_feat: A numpy array of (2048, )
                local_feat: A dict containing both locations and descriptors.
                            For example: local_feat['locations'] and
                            local_feat['descriptors']
                bbox_info: A dict for basic image information including
                           height and width.
                bbox: bbox for query
        """

        _ = self._cfg
        _, _ = large_path, bbox
        res = {'global_feat': [], 'local_feat': [],
               'bbox_info': [], 'bbox': []}
        return res

    def local_matching(self, query_res, index_res):
        """Sample-to-sample local feature matching.

        Args:
            query_res: A dict for query feat.
            index_res: A dict for index feat.

        Returns:
            Matching results in detection format. For example:
            [[0, 0, 100, 100, 0.5], [10, 10, 90, 90, 0.7]]
        """

        _ = self._cfg
        _, _ = query_res, index_res
        dets = []
        return dets

    def _get_ignore_dic(self):
        """Gets ignore dic for each query_instance_id, with only running once.

        Returns:
            A dict to store inogre image list for each object_id.
            For example: {'ojbect_1': [aaa.jpg, bbg.jpg, ccc.jpg],
                          'object_2': [ccc.jpg, fff.jpg]}
        """

        ignore_dic = {}
        query_instance_dic = pickle.load(open(self._cfg.EVAL.LABEL_FILE, 'rb'))
        for object_id, query_info in query_instance_dic.items():
            ignore_list = []
            for object_data in query_info['pos_gallery_list']:
                image_name = object_data['image_name']
                if 'ignore' in object_data:
                    if object_data['ignore']:
                        ignore_list.append(image_name)
            ignore_dic[object_id] = ignore_list
        return ignore_dic

    def get_global_rank(self, query_sim, all_index_res):
        """Gets global rank for a particular query.

        Args:
            query_sim: gallery sim to a specific query
            all_index_res: all index feature

        Returns:
            A dict to contain global ranking dets for each large image.
            For example: {'aaa.jpg': [[0, 0, 100, 100, 0.4],
                          'bbb.jpg': [[10, 10, 90, 90, 0.5]]}
        """

        # sample_query_global_feat = self.query_res['global_feat']
        # if len(sample_query_global_feat) == 0:
        #     print('find empty query, use rand feature insead')
        #     sample_query_global_feat = np.random.rand((2048))
        index_name_list = list(all_index_res.keys())
        init_orders = np.argsort(-query_sim)
        global_rank_dict = OrderedDict()
        for rank_index in init_orders:
            if len(global_rank_dict) >= self._cfg.EVAL.RANK_GLOBAL_TOPK[-1]:
                continue
            index_name = index_name_list[rank_index]
            # TODO, for xianzhe comment the two lines
            if index_name in self.ignore_data:
                continue
            sim = query_sim[rank_index]
            merged_bbox = np.array([[0, 0, 0, 0, sim]])
            global_rank_dict[index_name] = merged_bbox
        return global_rank_dict

    def rerank_by_local(self, global_rank_dict, all_index_res):
        """Reranks by local features for one query only. Learn from DELG.

        Args:
            global_rank_dict: A dict containing global rank result
            all_index_res: A dict containing index feat.

        Returns:
            A dict containing local rank result
        """

        self.rerank_num = self._cfg.DELG.NUM_TO_RERANK
        if self.rerank_num > 1000:
            self.rerank_num = len(global_rank_dict)
        index_name_list = list(global_rank_dict.keys())[:self.rerank_num]
        if self._cfg.EVAL.REFINER == 'delg_vit':
            local_rank_dict = self.local_matching_batch(self.query_res, index_name_list)
        else:
            local_rank_dict = self.local_matching_batch(all_index_res, index_name_list)

        local_rank_dict = self._pad_local_dets(local_rank_dict,
                                               global_rank_dict)
        return local_rank_dict

    def _pad_local_dets(self, local_rank_dict, global_rank_dict):
        """Pads local dets to have the same lenth as global rank.

        For the data not in reranking, we just copy the bbox from global rank.

        Args:
            local_rank_dict: A dict containing global local result.
            global_rank_dict: A dict containing global rank result

        Returns:
            A dict containing local ranking dets for each large image.
        """

        merged_rank_dict = global_rank_dict.copy()
        ordered_index_name_list = list(global_rank_dict.keys())
        for global_rank in range(self.rerank_num):
            index_name = ordered_index_name_list[global_rank]
            merged_rank_dict[index_name] = local_rank_dict[index_name]
        return merged_rank_dict

    def merge_ranks(self, local_rank_dict, global_rank_dict):
        """Merges local rank results into global ranking results.

        It is very important that we reorder index names and sims after
        global ranking. Or we may have confused results.

        Args:
            local_rank_dict: A dict containing local rank result
            global_rank_dict: A dict containing global rank result

        Returns:
            A dict containing global+local ranking dets for each large image.
        """

        def _InliersInitialScoresSorting(k):    # pylint: disable=invalid-name
            """Helper function to sort list based on two entries."""

            # pylint: disable=line-too-long
            # sorted by two key, ref: https://stackoverflow.com/questions/4233476/sort-a-list-by-multiple-attributes # noqa
            return (merged_score[k][0], merged_score[k][1])

        global_dets = np.array([item[0] for item in global_rank_dict.values()])
        local_dets = np.array([item[0] for item in local_rank_dict.values()])
        ordered_index_name_list = list(global_rank_dict.keys())

        merged_score = np.concatenate(
                [local_dets[:, 4:5], global_dets[:, 4:5]], axis=1)
        output_ranks = sorted(range(len(global_rank_dict)),
                              key=_InliersInitialScoresSorting,
                              reverse=True)
        local_rank_dict = OrderedDict()
        for rank, order in enumerate(output_ranks):
            dets = local_dets[order]
            dets[-1] = -rank
            index_name = ordered_index_name_list[order]
            local_rank_dict[index_name] = dets[np.newaxis, :]
        return local_rank_dict
