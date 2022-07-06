# -*- coding:utf-8 -*-
"""Base refiner which could refine results by localizer.
Author: gongyou.zyq
Date: 2020.11.18
"""

class BaseRefiner:
    """Base refiner for all refiner."""

    def __init__(self, gpus, _cfg):

        self._cfg = _cfg
        _ = gpus
        self.query_feature_dic = {}

    def set_query(self, query_img_bgr, query_bbox, query_instance_id):
        """Set query for the model.

        Args:
            query_img_bgr: A opencv numpy of image.
            query_bbox: A numpy of bbox shape (4,)
            query_instance_id: A unique index for each query search sample.
        """

        _ = self._cfg
        _, _, _ = query_img_bgr, query_bbox, query_instance_id
        return self.query_feature_dic

    def set_query_feat(self, all_query_reid_feat):
        """Set all query from external source."""

        self.query_feature_dic = all_query_reid_feat

    def reset(self):
        """Reset cached query data.

        For multi query cases, we must manage the query data.
        """
        self.query_feature_dic = {}
