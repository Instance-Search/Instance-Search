# encoding: utf-8
"""
ONNX inference
"""

import os
import math

import numpy as np
import cv2
import onnxruntime as rt

from instance_search.base_refiner import BaseRefiner

# pylint: disable=too-many-instance-attributes
class ReIDInference(BaseRefiner):
    """ReID Inference."""

    def __init__(self, gpus, _cfg):

        BaseRefiner.__init__(self, gpus, _cfg)

        model_path = _cfg.INFERENCE.REID_MODEL
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
        self.sess = rt.InferenceSession(model_path)

        self.input_name = self.sess.get_inputs()[0].name
        # print("input name", input_name)
        self.output_name = self.sess.get_outputs()[0].name
        # print("output name", output_name)
        self.image_height = _cfg.INFERENCE.REID_IMAGE_HEIGHT
        self.image_width = _cfg.INFERENCE.REID_IMAGE_WIDTH
        self.mean = np.asarray([123.675, 116.280, 103.530])
        self.std = np.asarray([57.0, 57.0, 57.0])
        self.half_flag = _cfg.INFERENCE.REID_HALF_FLAG
        self.batch_size = _cfg.INFERENCE.REID_BATCH_SIZE    # tuned

    def set_query(self, query_img_bgr, query_bbox, query_instance_id):
        """Set query."""

        # pylint: disable=invalid-name
        [x1, y1, x2, y2] = query_bbox
        height, width = query_img_bgr.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2-1)
        y2 = min(height, y2-1)
        query_feature = self.extract(query_img_bgr[y1:y2, x1:x2, :])
        self.query_feature_dic[query_instance_id] = query_feature
        return self.query_feature_dic

    def extract(self, img):
        """Extract single small image feature without batch mode."""

        img = cv2.resize(img, (self.image_width, self.image_height))
        img = img[:, :, ::-1]
        img = (img-self.mean)/self.std
        img = img[np.newaxis, :, :, :]
        img = img.transpose((0, 3, 1, 2))
        if not self.half_flag:
            img = img.astype(np.float32)
        else:
            img = img.astype(np.float16)
        res = self.sess.run([self.output_name], {self.input_name: img})
        feat = np.array(res).flatten()
        if self._cfg.INFERENCE.REID_FLIP:
            flip_img = img[:, :, :, ::-1]
            res = self.sess.run([self.output_name], {self.input_name: flip_img})
            flip_feat = np.array(res).flatten()
            feat = (feat+flip_feat)/2.0
        feat = feat/np.linalg.norm(feat, ord=2, axis=0)

        return feat

    def extract_batch(self, img_list):
        """Extract multiple small image feature with batch mode."""

        feat_list = []
        num_batches = math.ceil(len(img_list) / self.batch_size)
        for batch_index in range(num_batches):
            batch_data = []
            for img in img_list[batch_index*self.batch_size:\
                                (batch_index+1)*self.batch_size]:
                img = cv2.resize(img, (self.image_width, self.image_height))
                batch_data.append(img)

            batch_data = np.array(batch_data)
            batch_data = batch_data[:, :, :, ::-1]
            batch_data = (batch_data-self.mean)/self.std
            batch_data = batch_data.transpose((0, 3, 1, 2))
            if not self.half_flag:
                batch_data = batch_data.astype(np.float32)
            else:
                batch_data = batch_data.astype(np.float16)
            res = self.sess.run([self.output_name],
                                {self.input_name: batch_data})
            feat = np.array(res)[0]
            feat = feat/np.linalg.norm(feat, ord=2, axis=1, keepdims=True)
            feat_list.append(feat)

        if len(feat_list) > 0:
            feat_list = np.vstack(feat_list)
        else:
            feat_list = np.array([])
        return feat_list

    def extract_bboxes_nobatch(self, large_img, bboxes):
        """Extract multiple bboxes feature in one large image without batch mode."""

        feat_list = []
        for bbox in bboxes:
            # pylint: disable=invalid-name
            [x1, y1, x2, y2] = bbox.astype('int')
            patch = large_img[y1:y2, x1:x2, :]
            patch = cv2.resize(patch, (self.image_width, self.image_height))
            patch = patch[:, :, ::-1]
            patch = (patch-self.mean)/self.std
            patch = patch[np.newaxis, :, :, :]
            patch = patch.transpose((0, 3, 1, 2))
            if not self.half_flag:
                patch = patch.astype(np.float32)
            else:
                patch = patch.astype(np.float16)
            res = self.sess.run([self.output_name], {self.input_name: patch})
            feat = np.array(res).flatten()
            if self._cfg.INFERENCE.REID_FLIP:
                flip_patch = patch[:, :, :, ::-1]
                res = self.sess.run([self.output_name], {self.input_name: flip_patch})
                flip_feat = np.array(res).flatten()
                feat = (feat+flip_feat)/2.0
            feat = feat/np.linalg.norm(feat, ord=2, axis=0)
            feat_list.append(feat)

        if len(feat_list) > 0:
            feat_list = np.vstack(feat_list)
        else:
            feat_list = np.array([])
        return feat_list

    def extract_bboxes_batch(self, large_img, bboxes):
        """Extract multiple bboxes feature in one large image with batch mode."""

        feat_list = []
        num_batches = math.ceil(len(bboxes) / self.batch_size)
        # print(num_batches)
        for batch_index in range(num_batches):
            batch_data = []
            for bbox in bboxes[batch_index*self.batch_size:
                               (batch_index+1)*self.batch_size]:
                # pylint: disable=invalid-name
                [x1, y1, x2, y2] = bbox.astype('int')
                # print(x1, y1, x2, y2)
                patch = large_img[y1:y2, x1:x2, :]
                patch = cv2.resize(patch, (self.image_width, self.image_height))
                batch_data.append(patch)

            batch_data = np.array(batch_data)
            batch_data = batch_data[:, :, :, ::-1]
            batch_data = (batch_data-self.mean)/self.std
            batch_data = batch_data.transpose((0, 3, 1, 2))
            if not self.half_flag:
                batch_data = batch_data.astype(np.float32)
            else:
                batch_data = batch_data.astype(np.float16)
            res = self.sess.run([self.output_name],
                                {self.input_name: batch_data})
            feat = np.array(res)[0]
            if self._cfg.INFERENCE.REID_FLIP:
                flip_data = batch_data[:, :, :, ::-1]
                res = self.sess.run([self.output_name], {self.input_name: flip_data})
                flip_feat = np.array(res)[0]
                feat = (feat+flip_feat)/2.0
            feat = feat/np.linalg.norm(feat, ord=2, axis=1, keepdims=True)
            feat_list.append(feat)
        if len(feat_list) > 0:
            feat_list = np.vstack(feat_list)
        else:
            feat_list = np.array([])
        return feat_list

    def extract_bboxes(self, large_img, bboxes):
        """For external use only."""

        if self.batch_size > 1:
            feat_list = self.extract_bboxes_batch(large_img, bboxes)
        else:
            feat_list = self.extract_bboxes_nobatch(large_img, bboxes)
        return feat_list

    @staticmethod
    def distance(feat1, feat2):
        """Compute distance."""

        sim = np.linalg.norm(feat1-feat2, ord=2)
        return sim
