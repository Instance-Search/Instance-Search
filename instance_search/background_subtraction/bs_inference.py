"""
Background subtraction for moving object detection
Author: gongyou.zyq
Date: 2021.02.07
Ref: https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
     https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
     https://zhuanlan.zhihu.com/p/42944850
"""

import os
import numpy as np
import cv2

from instance_search.base_localizer import BaseLocalizer

class BackgroundSubtraction(BaseLocalizer):
    """
    Background subtraction for region proposals.
    """

    def __init__(self, gpus, _cfg):

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
        BaseLocalizer.__init__(self, gpus, _cfg)
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.scale = 0.25
        self.top_proposal = self._cfg.INFERENCE.LOCALIZER_TOPK
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    def reset(self):
        """For new device_id."""

        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    # pylint: disable=arguments-differ
    def detect(self, img_bgr):
        """Detect image."""

        img_bgr = cv2.resize(img_bgr, dsize=(0, 0), fx=self.scale, fy=self.scale)
        fgmask = self.fgbg.apply(img_bgr)
        draw1 = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
        draw1 = cv2.dilate(draw1, self.kernel, iterations=1)
        contours_m, _ = cv2.findContours(draw1.copy(),\
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = []
        for cnt in contours_m:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            rects = cv2.boundingRect(cnt)
            rects = list(rects)
            rects[2] += rects[0]
            rects[3] += rects[1]
            fg_region = fgmask[rects[1]: rects[3], rects[0]: rects[2]].copy()
            score = np.mean(fg_region) / 255.0
            rects.append(score)
            new_bbox = np.array(rects)
            new_bbox[:4] = new_bbox[:4] / self.scale
            result.append(new_bbox)
        return np.array(result)[:self.top_proposal]
