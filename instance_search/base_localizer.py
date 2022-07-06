# -*- coding:utf-8 -*-
"""Base localizer which could serve as rough detector.
Author: gongyou.zyq
Date: 2020.11.18
"""

import os
import cv2
import numpy as np
import torch

class BaseLocalizer:
    """Base localizer for all rough localizer."""

    def __init__(self, gpus, _cfg):

        self._cfg = _cfg
        _ = gpus
        self.bbox_feats_z_dic = {}
        self.activated_query_list = []

    def set_activated_query(self, activated_query_list):
        """Some query may be not responsible for a gallery and thus we skip."""

        self.activated_query_list = activated_query_list

    def set_query(self, query_img_bgr, query_bbox, query_instance_id):
        """Sets query for the model.

        Args:
            query_img_bgr: A opencv numpy of image.
            query_bbox: A numpy of bbox shape (4,)
        """

        _ = self._cfg
        _, _, _ = query_img_bgr, query_bbox, query_instance_id
        return self.bbox_feats_z_dic

    def set_query_feat(self, all_query_loc_feat):
        """Sets query from external source."""

        for query_instance_id, query_feat in all_query_loc_feat.items():
            all_query_loc_feat[query_instance_id] = torch.tensor(query_feat).cuda()
        self.bbox_feats_z_dic = all_query_loc_feat

    def reset(self):
        """Reset cached query data.

        For multi query cases, we must manage the query data carefully.
        """

        self.bbox_feats_z_dic = {}
        self.activated_query_list = []

    def detect(self, img_bgr_list):
        """Detect with given image after query is set.

        Args:
            img_bgr_list: A list of image numpy with shape[h, w, c]

        Returns:
            Numpy array of shape [N, 5], where N is bbox num,
            first four elements are bbox and last value is score.
            For example: [[100, 100, 200, 200, 0.4],
                          [30, 30, 40, 40, 0.7]]
            If no bboxes detected, empty list is returned.
        """

        _ = self._cfg
        _ = img_bgr_list

    @staticmethod
    def _draw_bboxes(query_info, gallery_info):
        """Draw bboxes on query and gallery images."""

        [query_large_path, query_bbox] = query_info
        [gallery_large_path, det_result] = gallery_info
        query_img = cv2.imread(query_large_path)
        [x1, y1, x2, y2] = query_bbox    # pylint: disable=invalid-name
        cv2.rectangle(query_img, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 0, 255), 2)
        cv2.putText(query_img, 'query', (int(x1), int(y1)),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

        orders = np.argsort(-det_result[:, -1])
        gallery_img = cv2.imread(gallery_large_path)
        print(f'{len(det_result)} detections')
        for index in orders[:3]:
            # pylint: disable=invalid-name
            [x1, y1, x2, y2, score] = det_result[index]
            print(det_result[index])
            cv2.putText(gallery_img, '%.2f' % score, (int(x1), int(y1)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(gallery_img, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
        return query_img, gallery_img

    def draw(self, query_info, gallery_info, save_name):
        """Draw query bbox+gallery detection results on a pair of images.

        Draw query and gallery bboxes individually and then resize them
        to a fixed height. Then merge the two images and save for debug.

        Args:
            query_info: Sigle query image path and bbox.
            gallery_info: Single gallery image path and detection results.
            save_name: Path to save the result image.
        Raises:
            null
        """

        query_img, gallery_img = self._draw_bboxes(query_info, gallery_info)
        fixed_height = max(query_img.shape[0], gallery_img.shape[0])
        if query_img.shape[0] != fixed_height:
            ratio = float(fixed_height) / query_img.shape[0]
            query_img = cv2.resize(query_img, (0, 0), fx=ratio, fy=ratio)
        if gallery_img.shape[0] != fixed_height:
            ratio = float(fixed_height) / gallery_img.shape[0]
            gallery_img = cv2.resize(gallery_img, (0, 0), fx=ratio, fy=ratio)
        merged_img = np.concatenate([query_img, gallery_img], axis=1)
        save_dir = os.path.dirname(save_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(save_name, merged_img)
