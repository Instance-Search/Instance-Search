"""Sliding window inference.
Author: gongyou.zyq
Date: 2020.11.19
"""

from math import sqrt, ceil
import numpy as np

from instance_search.base_localizer import BaseLocalizer

class FullBbox(BaseLocalizer):
    """FullBbox."""

    def __init__(self, gpus, _cfg):
        BaseLocalizer.__init__(self, gpus, _cfg)
        # since test image size is not fixed, we cannot share detection
        # self.detection = np.load('./instance_search/sliding_window/cache.npy')
        self.query_num = 0

    def set_query(self, large_img_bgr, query_bbox, query_instance_id):
        """Set query."""

        self.query_num += 1
        return {}

    def reset(self):
        """Reset."""

        self.query_num = 0

    def detect(self, img_bgr_list):
        """Detect sliding window."""

        result = []
        for img_bgr in img_bgr_list:
            per_result = []
            raw_height, raw_width = img_bgr.shape[:2]
            per_result = [[0, 0, raw_width-1, raw_height-1, 1.0]]
            result.append(np.array(per_result).astype('float32'))
        det_queries = {query_instance_id: result for query_instance_id in\
                       self.activated_query_list}
        return det_queries
