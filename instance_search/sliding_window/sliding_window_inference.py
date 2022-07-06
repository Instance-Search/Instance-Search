"""Sliding window inference.
Author: gongyou.zyq
Date: 2020.11.19
"""

from math import sqrt, ceil
import numpy as np

from instance_search.base_localizer import BaseLocalizer

class SlidingWindow(BaseLocalizer):
    """SlidingWindow."""

    def __init__(self, gpus, _cfg):
        BaseLocalizer.__init__(self, gpus, _cfg)
        # since test image size is not fixed, we cannot share detection
        # self.detection = np.load('./instance_search/sliding_window/cache.npy')
        self.query_num = 0

    def set_query(self, large_img_bgr, query_bbox, query_instance_id):
        """Set query."""

        self.query_num += 1

    def reset(self):
        """Reset."""

        self.query_num = 0

    def detect(self, img_bgr_list):
        """Detect sliding window."""

        result = []
        for img_bgr in img_bgr_list:
            per_result = []
            raw_height, raw_width = img_bgr.shape[:2]
            scale_list = [70, 140, 210, 280, 350, 420]
            for ratio in [0.5, 1.0, 2.0]:
                for scale in scale_list:
                    height = scale * sqrt(ratio)
                    width = scale / sqrt(ratio)
                    step_y = int(height * 1.0)
                    step_x = int(width * 1.0)
                    for x_index in range(ceil(raw_width/float(step_x))):
                        for y_index in range(ceil(raw_height/float(step_y))):
                            bbox = [x_index*step_x, y_index*step_y,
                                    x_index*step_x+width,
                                    y_index*step_y+height, 1.0]
                            per_result.append(bbox)
            result.append(np.array(per_result).astype('float32'))
        # print(f'extract time {time.time()-start}')
        # print(results.shape)
        # np.save('./instance_search/sliding_window/cache', results)
        det_queries = {query_instance_id: result for query_instance_id in\
                       self.activated_query_list}
        return det_queries
