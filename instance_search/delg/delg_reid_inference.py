"""Extract global-reid features in the same format as delg.
Author: gongyou.zyq
Date: 2021.06.10
"""

import cv2
import numpy as np

from instance_search.base_matcher import BaseMatcher
from instance_search.reid.reid_inference import ReIDInference


class DelgReIDMatcher(BaseMatcher):
    """Delg global local inference."""

    def __init__(self, gpus, _cfg):

        BaseMatcher.__init__(self, gpus, _cfg)
        if gpus is not None:
            self.global_reid = ReIDInference(gpus, _cfg)

    def extract(self, large_path, bbox):
        """We share the extracted feature from DELG"""

        large_img = cv2.imread(large_path)
        # vertical_flag = self.check_vertical(large_img)
        # if not vertical_flag:
        #     # print(large_path)
        #     large_img = cv2.transpose(large_img)
        if bbox is None:
            global_feat = self.global_reid.extract(large_img)
        else:
            bbox = np.array([bbox])
            global_feat = self.global_reid.extract_bboxes_nobatch(large_img,
                                                                  bbox)
            bbox = bbox[0].astype('int')
        global_feat = global_feat.flatten()
        height, width = large_img.shape[:2]
        bbox_info = {'w': width, 'h': height}
        res = {'global_feat': global_feat, 'local_feat': [],
               'bbox_info': bbox_info, 'bbox': bbox}
        return res

    def check_vertical(self, img):
        """Check vertical or horizontal.

        The tricks work for /home/gongyou.zyq/datasets/instance_search/GLDv2/reid_images/test_gallery/0000_0008_0008.jpg      
        """

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
        vertical = cv2.Sobel(gray, -1, 1, 0)
        horizontal = cv2.Sobel(gray, -1, 0, 1)
        ratio = np.sum(np.abs(horizontal)) / np.sum(np.abs(vertical))
        return ratio < 2.0
