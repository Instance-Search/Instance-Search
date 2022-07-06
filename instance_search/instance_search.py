"""Combine localizer+refiner together as InstanceSearcher.
Author: gongyou.zyq
Date: 2020.11.28
"""

import numpy as np

from instance_search.sliding_window.sliding_window_inference import SlidingWindow
from instance_search.globaltracker.globaltracker_inference import GlobalTracker
# from instance_search.siamrpn_tracker.siamese_inference import SiameseExtractor
# from instance_search.transt.transt_inference import TransTracker
from instance_search.yolo_detector.yolo_detector import YoloDetector
from instance_search.selective_search.selective_search_inference import SelectiveSearch
from instance_search.edge_box.edge_box_inference import EdgeBox
from instance_search.full_bbox.full_bbox_inference import FullBbox
from instance_search.reid.reid_inference import ReIDInference
from instance_search.base_localizer import BaseLocalizer
from instance_search.base_refiner import BaseRefiner


def localizer_factory(gpu_id, _cfg):
    """Rough Localizer for multiple kinds.

    We supoort sliding_window, globaltrack, siamrpn and general_detection, etc.

    Args:
        gpu_id: A int indicating which gpu to use
        _cfg: Instance search config.

    Returns:
        Localizer object.
    """

    if _cfg.EVAL.ROUGH_LOCALIZER == 'sliding_window':
        _localizer = SlidingWindow(gpu_id, _cfg)
    elif _cfg.EVAL.ROUGH_LOCALIZER == 'globaltrack':
        _localizer = GlobalTracker(gpu_id, _cfg)
    elif (_cfg.EVAL.ROUGH_LOCALIZER == 'siamrpn' or
            _cfg.EVAL.ROUGH_LOCALIZER == 'siamrpn_full'):
        _localizer = SiameseExtractor(gpu_id, _cfg)
    elif 'transt' in _cfg.EVAL.ROUGH_LOCALIZER:
        _localizer = TransTracker(gpu_id, _cfg)
    elif _cfg.EVAL.ROUGH_LOCALIZER == 'general_detection':
        _localizer = YoloDetector(gpu_id, _cfg)
    elif _cfg.EVAL.ROUGH_LOCALIZER == 'selective_search':
        _localizer = SelectiveSearch(gpu_id, _cfg)
    elif _cfg.EVAL.ROUGH_LOCALIZER == 'edge_box':
        _localizer = EdgeBox(gpu_id, _cfg)
    elif _cfg.EVAL.ROUGH_LOCALIZER == 'full_bbox':
        _localizer = FullBbox(gpu_id, _cfg)
    else:
        print('unknown localizer.')
        _localizer = None
    return _localizer


def refine_factory(gpu_id, _cfg):
    """Refine mdoel for multiple kinds.

    Args:
        gpu_id: A int indicating which gpu to use
        _cfg: Instance search config.

    Returns:
        Refiner object.
    """

    if _cfg.EVAL.REFINER == 'reid':
        _refiner = ReIDInference(gpu_id, _cfg)
    elif _cfg.EVAL.REFINER == 'null':
        _refiner = BaseRefiner(gpu_id, _cfg)
    else:
        print('unknown refiner.')
        _refiner = None
    return _refiner


class InstanceSearcher(BaseLocalizer):
    """Instance Searcher."""

    def __init__(self, gpu_id, _cfg):
        BaseLocalizer.__init__(self, gpu_id, _cfg)
        self.localizer = localizer_factory(gpu_id, _cfg)
        self.refiner = refine_factory(gpu_id, _cfg)
        self.query_id_list = []
        self.valid_query_id_list = []
        shared_loc_type = ['edge_box', 'general_detection', 'selective_search',
                           'full_bbox']
        self.shared_det_flag = _cfg.EVAL.ROUGH_LOCALIZER in shared_loc_type

    # def get_query_num(self):
    #     """Get query number."""
    #
    #     return len(self.query_id_list)

    def set_query(self, query_img_bgr, query_bbox, query_instance_id):
        """Set query. See base class"""

        query_loc_feat_dic = self.localizer.set_query(query_img_bgr, query_bbox,
                                                 query_instance_id)
        query_reid_feat_dic = self.refiner.set_query(query_img_bgr, query_bbox,
                                                query_instance_id)
        # query-guided localizer
        if len(query_loc_feat_dic) > 0 and len(query_reid_feat_dic) > 0:
            return [{query_instance_id: query_loc_feat_dic[query_instance_id]},
                    {query_instance_id: query_reid_feat_dic[query_instance_id]}]
        elif len(query_loc_feat_dic) > 0 and len(query_reid_feat_dic) == 0:
            return [{query_instance_id: query_loc_feat_dic[query_instance_id]},
                    {}]
        elif len(query_loc_feat_dic) == 0 and len(query_reid_feat_dic) > 0:
            return [{},
                    {query_instance_id: query_reid_feat_dic[query_instance_id]}]
        else:
            return [{}, {}]

    def set_query_feat(self, all_query_loc_feat, all_query_reid_feat):
        """Set query feat by external source."""

        self.localizer.set_query_feat(all_query_loc_feat)
        self.refiner.set_query_feat(all_query_reid_feat)

    def reset(self):
        """Reset globaltrack + reid for each worker."""

        self.localizer.reset()
        self.refiner.reset()

    def detect(self, img_bgr_list):
        """Uses localizer as candidate proposals and uses reid for refinement.

        Args:
            img_bgr: A numpy opencv image.
        Returns:
            A 2-element tuple:
                rough_det: Rough detection of shape (N, 5)
                refine_det: Refined detection of shape (N, 5)
        """

        activated_query_list = self.activated_query_list
        self.localizer.set_activated_query(activated_query_list)

        rough_det_queries = self.localizer.detect(img_bgr_list)
        if self._cfg.EVAL.REFINER == 'null':
            refined_det_queries = {query_instance_id: [[]*len(img_bgr_list)]
                                   for query_instance_id in rough_det_queries}
        elif self._cfg.EVAL.MULTI_QUERY or self.shared_det_flag:
            refined_det_queries = self.refine_multi_query(rough_det_queries,
                                                          img_bgr_list)
        else:
            refined_det_queries = self.refine_single_query(rough_det_queries,
                                                           img_bgr_list)
        return rough_det_queries, refined_det_queries

    def refine_multi_query(self, rough_det_queries, img_bgr_list):
        """Refine multi query."""

        # NOTE: trick for speed, only one gallery per time
        refined_det_queries = {}
        shared_gallery_feat = None    # shared for all gallery
        # loop over activated queries
        for query_instance_id, rough_det_list in rough_det_queries.items():
            query_feature = self.refiner.query_feature_dic[query_instance_id]
            refined_det_list = []
            for j, rough_det in enumerate(rough_det_list):
                refined_det = []
                if len(rough_det) == 0:
                    refined_det_list.append(refined_det)
                    continue
                good_bboxes = rough_det[:, :4]
                if shared_gallery_feat is None:
                    shared_gallery_feat = self.refiner.extract_bboxes(
                            img_bgr_list[j], good_bboxes)
                feat_array = shared_gallery_feat
                new_sim_array = np.matmul(feat_array,
                                          query_feature[:, np.newaxis])
                refined_det = np.concatenate([good_bboxes, new_sim_array],
                                             axis=1)
                refined_det = refined_det.astype('float16')
                refined_det_list.append(refined_det)
            refined_det_queries[query_instance_id] = refined_det_list
        return refined_det_queries

    def refine_single_query(self, rough_det_queries, img_bgr_list):
        """Refine single query."""

        refined_det_queries = {}
        # loop over activated queries
        for query_instance_id, rough_det_list in rough_det_queries.items():
            query_feature = self.refiner.query_feature_dic[query_instance_id]
            refined_det_list = []
            for j, rough_det in enumerate(rough_det_list):
                refined_det = []
                if len(rough_det) == 0:
                    refined_det_list.append(refined_det)
                    continue
                good_bboxes = rough_det[:, :4]
                feat_array = self.refiner.extract_bboxes(img_bgr_list[j],
                                                         good_bboxes)
                new_sim_array = np.matmul(feat_array,
                                          query_feature[:, np.newaxis])
                refined_det = np.concatenate([good_bboxes, new_sim_array],
                                             axis=1)
                refined_det = refined_det.astype('float16')
                refined_det_list.append(refined_det)
            refined_det_queries[query_instance_id] = refined_det_list
        return refined_det_queries

    def reid_rerank(self, rough_det_queries, img_bgr_list):
        """Uses localizer proposals as input and uses reid for refinement.

        Args:
            img_bgr: A numpy opencv image.
        Returns:
            A 1-element tuple:
                refine_det: Refined detection of shape (N, 5)
        """

        assert self._cfg.EVAL.REFINER != 'null'
        if self._cfg.EVAL.MULTI_QUERY or self.shared_det_flag:
            refined_det_queries = self.refine_multi_query(rough_det_queries,
                                                          img_bgr_list)
        else:
            refined_det_queries = self.refine_single_query(rough_det_queries,
                                                           img_bgr_list)
        return refined_det_queries
