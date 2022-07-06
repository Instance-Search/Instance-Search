"""
Selective search for region proposals
Author: gongyou.zyq
Date: 2021.02.07
Ref: https://www.pyimagesearch.com/2020/06/29/opencv-selective-search-for-object-detection/
"""

import os
import numpy as np
import cv2

from instance_search.base_localizer import BaseLocalizer

class SelectiveSearch(BaseLocalizer):
    """
    Selective search for region proposals.
    """

    def __init__(self, gpus, _cfg):

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
        BaseLocalizer.__init__(self, gpus, _cfg)
        self.ss_model = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.scale = 0.25
        self.top_proposal = 20

    def detect(self, img_bgr_list):
        """Detect image."""

        result = []
        for img_bgr in img_bgr_list:
            img_bgr = cv2.resize(img_bgr, dsize=(0, 0), fx=self.scale, fy=self.scale)
            self.ss_model.setBaseImage(img_bgr)
            self.ss_model.switchToSelectiveSearchFast()
            rects = self.ss_model.process()
            rects = rects[:self.top_proposal] / self.scale
            rects[:, 2] += rects[:, 0]
            rects[:, 3] += rects[:, 1]
            score = np.linspace(1.0, 1.0/self.top_proposal, len(rects))[:, np.newaxis]
            new_bbox = np.concatenate([rects, score], axis=1)
            result.append(new_bbox)
        det_queries = {query_instance_id: result for query_instance_id in\
                self.activated_query_list}
        return det_queries
