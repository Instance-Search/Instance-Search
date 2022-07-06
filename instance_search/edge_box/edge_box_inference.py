"""
Edge box for region proposals
Author: gongyou.zyq
Date: 2021.02.07
Ref: https://stackoverflow.com/questions/54843550/edge-box-detection-using-opencv-python
     https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/samples/edgeboxes_demo.py
"""

import os
import numpy as np
import cv2

from instance_search.base_localizer import BaseLocalizer

class EdgeBox(BaseLocalizer):
    """
    Edge box for region proposals.
    """

    def __init__(self, gpus, _cfg):

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
        BaseLocalizer.__init__(self, gpus, _cfg)
        model = 'instance_search/edge_box/model.yml'
        self.edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
        self.scale = 0.25
        self.top_proposal = self._cfg.INFERENCE.LOCALIZER_TOPK

    def detect(self, img_bgr_list):
        """Detect image."""

        result = []
        for img_bgr in img_bgr_list:
            img_bgr = cv2.resize(img_bgr, dsize=(0, 0), fx=self.scale, fy=self.scale)
            rgb_im = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            edges = self.edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
            orimap = self.edge_detection.computeOrientation(edges)
            edges = self.edge_detection.edgesNms(edges, orimap)
            edge_boxes = cv2.ximgproc.createEdgeBoxes()
            edge_boxes.setMaxBoxes(self.top_proposal)
            rects, scores = edge_boxes.getBoundingBoxes(edges, orimap)
            if len(rects) > 0:
                rects = rects / self.scale
                rects[:, 2] += rects[:, 0]
                rects[:, 3] += rects[:, 1]
                new_bbox = np.concatenate([rects, scores], axis=1)
            else:
                new_bbox = []
            result.append(new_bbox)
        det_queries = {query_instance_id: result for query_instance_id in\
                self.activated_query_list}
        return det_queries
