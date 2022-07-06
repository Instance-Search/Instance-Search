"""
Python 3 wrapper for identifying objects in images
Author: gongyou.zyq
Date: 2020.11.19
Ref: https://github.com/AlexeyAB/darknet/blob/master/darknet_video.py
     https://github.com/AlexeyAB/darknet/blob/master/darknet_images.py
"""

import os
import numpy as np
import cv2

import instance_search.yolo_detector.darknet as darknet
from instance_search.base_localizer import BaseLocalizer
from instance_search.config import cfg

class YoloDetector(BaseLocalizer):
    """
    Yolo detector for general/custom object.
    """

    def __init__(self, gpus, _cfg):

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
        BaseLocalizer.__init__(self, gpus, _cfg)

        config_file = "./instance_search/yolo_detector/cfg_ins/yolov4.cfg"
        data_file = "./instance_search/yolo_detector/cfg_ins/coco.data"
        weights = "./instance_search/yolo_detector/model_ins/yolov4.weights"
        self.network, self.class_names, _ = darknet.load_network(config_file, data_file,
                                                                 weights, batch_size=1)
        if _cfg.SELF_TRAIN.ONLY_PERSON:
            self.class_names = ['person']
        self.train_dir = './instance_search/yolo_detector/train_images'
        self.top_proposal = self._cfg.INFERENCE.LOCALIZER_TOPK

        # Darknet doesn't accept numpy images.
        # Create one with image we reuse for each detect
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        self.darknet_image = darknet.make_image(self.width, self.height, 3)

    @staticmethod
    def scale_bbox(bbox, ratio_x, ratio_y, confidence):
        """Scale darknet result into (x1, y1, x2, y2) in original space."""

        x1, y1, x2, y2 = darknet.bbox2points(bbox)    # pylint: disable=invalid-name
        new_bbox = np.array([x1/ratio_x, y1/ratio_y, x2/ratio_x, y2/ratio_y, float(confidence)])
        return new_bbox

    def detect(self, img_bgr_list):
        """Detect image."""

        result = []
        for img_bgr in img_bgr_list:
            height, width = img_bgr.shape[:2]
            image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (self.width, self.height),
                                       interpolation=cv2.INTER_LINEAR)
            ratio_x = self.width/float(image_rgb.shape[1])
            ratio_y = self.height/float(image_rgb.shape[0])

            darknet.copy_image_from_bytes(self.darknet_image, image_resized.tobytes())
            detections = darknet.detect_image(self.network, self.class_names, self.darknet_image,
                                              thresh=0.02, nms=0.3)
            # detections = darknet.detect_image(self.network, self.class_names, self.darknet_image,
            #                                   thresh=0.9, nms=0.3)
            # darknet.free_image(self.darknet_image)

            # image = cv2.imread(imagePath)
            # print("*** "+str(len(detections))+" Results, color coded by confidence ***")
            per_result = []
            for _, confidence, bbox in detections:
                new_bbox = self.scale_bbox(bbox, ratio_x, ratio_y, confidence)
                # bbox should not be too small
                if new_bbox[3] - new_bbox[1] < 10 or new_bbox[2] - new_bbox[0] < 10:
                    continue
                per_result.append(new_bbox)
            per_result = np.array(per_result)
            if len(per_result) > 0:
                indexes = np.argsort(-per_result[:, -1])[:self.top_proposal]
                bbox = per_result[indexes, :]
                bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]], 0, width)
                bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]], 0, height)
                per_result = bbox
            result.append(np.array(per_result))
        det_queries = {query_instance_id: result for query_instance_id in\
                self.activated_query_list}
        return det_queries

def main():
    """Main function."""

    cfg.merge_from_file('./instance_search/config/CUHK_SYSU.yml')
    cfg.INFERENCE.LOCALIZER_TOPK = 10
    cfg.freeze()

    # img = cv2.imread('instance_search/yolo_detector/darknet/data/dog.jpg')
    img = cv2.imread('/home/gongyou.zyq/datasets/instance_search/PRW/all_images/c6/s2_000343.jpg')
    yolo = YoloDetector(0, cfg)
    yolo.set_activated_query(['debug_yolo'])
    det = yolo.detect([img])
    bboxes = det['debug_yolo'][0]
    print(bboxes.astype('float'))
    for bbox in bboxes:
        # pylint: disable=invalid-name
        [x1, y1, x2, y2, _] = bbox.astype('int')
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite('debug_yolo.jpg', img)


if __name__ == "__main__":
    main()
