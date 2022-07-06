"""
Optical flow for moving object detection with fixed/moving camera.
Author: gongyou.zyq
Date: 2021.02.07
Ref: https://zhuanlan.zhihu.com/p/42942198
"""

import os
import numpy as np
import cv2

from instance_search.base_localizer import BaseLocalizer

class OpticalFlow(BaseLocalizer):
    """
    Optical flow for region proposals.
    """

    def __init__(self, gpus, _cfg):

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
        BaseLocalizer.__init__(self, gpus, _cfg)
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.scale = 0.25
        self.top_proposal = self._cfg.INFERENCE.LOCALIZER_TOPK
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.init = False
        self.prvs = None
        self.hsv = None

    def reset(self):
        self.init = False
        self.prvs = None
        self.hsv = None

    # pylint: disable=arguments-differ
    def detect(self, img_bgr):
        """Detect image."""

        img_bgr = cv2.resize(img_bgr, dsize=(0, 0), fx=self.scale, fy=self.scale)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if not self.init:
            self.prvs = img_gray
            self.hsv = np.zeros_like(img_bgr)
            self.hsv[..., 1] = 255
        _next = img_gray
        if not np.array_equal(self.prvs.shape, _next.shape):
            # for CUHK-SYSU, we have no camera and thus image size changes all the time.
            self.reset()
            self.prvs = img_gray
            self.hsv = np.zeros_like(img_bgr)
            self.hsv[..., 1] = 255
        flow = cv2.calcOpticalFlowFarneback(self.prvs, _next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self.hsv[..., 0] = ang * 180 / np.pi / 2
        self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
        draw = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        draw1 = cv2.morphologyEx(draw, cv2.MORPH_OPEN, self.kernel)
        draw1 = cv2.threshold(draw1, 25, 255, cv2.THRESH_BINARY)[1]
        contours_m, _ = cv2.findContours(draw1.copy(),\
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        det = self.post_process(contours_m, draw)
        self.init = True
        self.prvs = _next
        return det

    def post_process(self, contours_m, draw):
        """Post process to get bbox."""

        result = []
        for cnt in contours_m:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            rects = cv2.boundingRect(cnt)
            rects = list(rects)
            rects[2] += rects[0]
            rects[3] += rects[1]
            fg_region = draw[rects[1]: rects[3], rects[0]: rects[2]].copy()
            score = np.mean(fg_region) / 255.0
            rects.append(score)
            new_bbox = np.array(rects)
            new_bbox[:4] = new_bbox[:4] / self.scale
            result.append(new_bbox)
        return np.array(result)
