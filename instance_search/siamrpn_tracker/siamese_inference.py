"""SiamRPN tracker inference.
Author: gongyou.zyq
Date: 2020.11.19
"""

import os
import sys
import numpy as np
import torch
import torchvision
sys.path.append('./instance_search/siamrpn_tracker/pysot')

# pylint: disable=wrong-import-position, too-many-locals, too-many-instance-attributes
import pysot.core.config as siamese_cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker
from instance_search.base_localizer import BaseLocalizer

def py_cpu_nms(dets, thresh=0.3):    # pylint: disable=too-many-locals
    """
    Pure Python NMS baseline.
    https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """

    # pylint: disable=invalid-name
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep, :]

def py_gpu_nms(full_bboxes, threshold=0.3):
    """GPU nms seems slower."""

    # pylint: disable=not-callable
    full_bboxes = torch.tensor(full_bboxes, device='cuda')
    scores = full_bboxes[:, 4]
    bboxes = full_bboxes[:, :4]
    keep = torchvision.ops.nms(bboxes, scores, threshold)
    bboxes = full_bboxes[keep].cpu().numpy()
    return bboxes

class SiameseExtractor(BaseLocalizer, SiamRPNLTTracker):
    """Siamese Extractor."""

    def __init__(self, gpus, _cfg):

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
        tracker_cfg = siamese_cfg.cfg
        # pylint: disable=line-too-long
        tracker_cfg.merge_from_file('./instance_search/siamrpn_tracker/pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml')
        tracker_cfg.CUDA = torch.cuda.is_available() and tracker_cfg.CUDA
        device = torch.device("cuda" if tracker_cfg.CUDA else "cpu")
        self.tracker_cfg = tracker_cfg

        model = ModelBuilder()

        # load model
        snapshot = './instance_search/siamrpn_tracker/pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/model.pth'
        model.load_state_dict(torch.load(snapshot,
                                         map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(device)
        BaseLocalizer.__init__(self, gpus, _cfg)
        SiamRPNLTTracker.__init__(self, model)

        if _cfg.EVAL.ROUGH_LOCALIZER == 'siamrpn':
            self.spacetime_consistency = True
            # set True to find correct object, False is default setting for siamrpnlt
            self.longterm_state = False
        else:
            self.spacetime_consistency = False
        self.query_num = 0

    def set_query(self, query_img_bgr, query_bbox, query_instance_id):
        """Set query."""

        bbox = np.array([query_bbox[0], query_bbox[1],
                         query_bbox[2]-query_bbox[0],
                         query_bbox[3]-query_bbox[1]])

        # pylint: disable=attribute-defined-outside-init
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])
        # calculate channle average
        self.channel_average = np.mean(query_img_bgr, axis=(0, 1))
        # pylint: enable=attribute-defined-outside-init

        # calculate z crop size
        context_amount = self.tracker_cfg.TRACK.CONTEXT_AMOUNT
        w_z = self.size[0] + context_amount * np.sum(self.size)
        h_z = self.size[1] + context_amount * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # get crop, size [1, 3, 127, 127]
        z_crop = self.get_subwindow(query_img_bgr, self.center_pos,
                                    self.tracker_cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)    # this will set only one zf.
        # used_layers: [2, 3, 4]
        # list: torch.Size([1, 128, 7, 7]) torch.Size([1, 256, 7, 7])
        # torch.Size([1, 512, 7, 7])
        bbox_feats_z = self.model.zf
        # bbox_feats_z_half = [item.half() for item in bbox_feats_z]
        bbox_feats_z_half = [item.detach().cpu().numpy() for item in bbox_feats_z]
        self.bbox_feats_z_dic[query_instance_id] = bbox_feats_z_half
        self.query_num += 1

    def detect(self, img_bgr_list):
        """Two kinds of siamrpn."""

        if self.spacetime_consistency and not self._cfg.EVAL.MULTI_QUERY:
            det_queries = self.detect_raw(img_bgr_list)
        elif not self.spacetime_consistency and not self._cfg.EVAL.MULTI_QUERY:
            det_queries = self.detect_global(img_bgr_list)
        else:
            det_queries = self.detect_multi_query(img_bgr_list)
        return det_queries

    def _get_anchors(self, instance_size):
        """Get anchors."""

        score_size = (instance_size - self.tracker_cfg.TRACK.EXEMPLAR_SIZE) // \
            self.tracker_cfg.ANCHOR.STRIDE + 1 + self.tracker_cfg.TRACK.BASE_SIZE
        hanning = np.hanning(score_size)
        window = np.outer(hanning, hanning)
        window = np.tile(window.flatten(), self.anchor_num)
        anchors = self.generate_anchor(score_size)
        return anchors, window

    def _get_scale_z_global(self, instance_size, img_bgr):
        """Get scale z."""

        context_amount = self.tracker_cfg.TRACK.CONTEXT_AMOUNT
        w_z = self.size[0] + context_amount * np.sum(self.size)
        h_z = self.size[1] + context_amount * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.tracker_cfg.TRACK.EXEMPLAR_SIZE / s_z

        s_x_old = s_z * (instance_size / self.tracker_cfg.TRACK.EXEMPLAR_SIZE)
        # pylint: disable=attribute-defined-outside-init
        s_x = np.max(img_bgr.shape[:2])
        temp_ratio = s_x / s_x_old

        self.center_pos = np.array([img_bgr.shape[1]/2.0, img_bgr.shape[0]/2.0])
        # Ref: https://blog.csdn.net/qq_33511693/article/details/90521161
        x_crop = self.get_subwindow(img_bgr, self.center_pos, instance_size,
                                    round(s_x), self.channel_average)
        # # this saves x_crop for vis purpose
        # temp = x_crop.cpu().numpy().astype('uint8')[0]
        # temp = temp.transpose((1, 2, 0))
        # cv2.imwrite('./test_images/x_crop.jpg', temp)
        new_scale_z = scale_z / temp_ratio
        return new_scale_z, x_crop

    def _convert_detection_global(self, bboxes, confs, img_bgr):
        """Convert detection into standard format."""

        # pylint: disable=invalid-name
        topk = 100
        indexes = np.argsort(-confs)[:topk]
        bboxes = bboxes.transpose((1, 0))
        cx = bboxes[:, 0] + self.center_pos[0]
        cy = bboxes[:, 1] + self.center_pos[1]
        width = bboxes[:, 2]
        height = bboxes[:, 3]
        x1 = np.clip(cx - width / 2, 0, None)[:, np.newaxis]
        y1 = np.clip(cy - height / 2, 0, None)[:, np.newaxis]
        x2 = np.clip(cx + width / 2, None, img_bgr.shape[1])[:, np.newaxis]
        y2 = np.clip(cy + height / 2, None, img_bgr.shape[0])[:, np.newaxis]
        confs = confs[:, np.newaxis]
        det = np.concatenate([x1, y1, x2, y2, confs], axis=1).astype('float32')
        # pylint: enable=invalid-name
        return det[indexes, :]

    @staticmethod
    def _get_bbox_global(pred_bbox, score, window, scale_z):
        """Get best bbox global."""

        _ = window
        # topk = 2000
        # indexes = np.argsort(-score)[:topk]
        # bbox = pred_bbox[:, indexes] / scale_z
        # conf = score[indexes]
        bbox = pred_bbox / scale_z
        conf = score
        return bbox, conf

    def detect_global(self, img_bgr_list):
        """Detect global image without spacetime consistency."""

        rough_det_queries = {}
        for query_instance_id in self.activated_query_list:
            det_galleries = []
            bbox_feats_z_half = self.bbox_feats_z_dic[query_instance_id]
            self.model.zf = [item.float() for item in bbox_feats_z_half]
            for img_bgr in img_bgr_list:
                # cfg.TRACK.LOST_INSTANCE_SIZE = 831 the size is fixed and much larger
                # than 255 in normal short term tracking.
                # Ref: https://www.gitmemory.com/xxAna
                # cfg.TRACK.BASE_SIZE: receptive field Ref: https://www.jianshu.com/p/cb5703a0a89a
                # instance_size = self.tracker_cfg.TRACK.LOST_INSTANCE_SIZE
                # instance_size = 831 * 2    # to make large enough
                instance_size = 831    # to make large enough
                scale_z, x_crop = self._get_scale_z_global(instance_size, img_bgr)

                outputs = self.model.track(x_crop)
                score = self._convert_score(outputs['cls'])
                anchors, window = self._get_anchors(instance_size)

                pred_bbox = self._convert_bbox(outputs['loc'], anchors)
                bbox, conf = self._get_bbox_global(pred_bbox, score, window, scale_z)
                det = self._convert_detection_global(bbox, conf, img_bgr)
                det = py_cpu_nms(det)
                # det = py_gpu_nms(det)
                det_galleries.append(det)
            rough_det_queries[query_instance_id] = det_galleries
        return rough_det_queries

    def _get_scale_z_raw(self, instance_size, img_bgr):
        """Get scale z raw."""

        w_z = self.size[0] + self.tracker_cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + self.tracker_cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.tracker_cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (instance_size / self.tracker_cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img_bgr, self.center_pos, instance_size,
                                    round(s_x), self.channel_average)
        return scale_z, x_crop

    def _convert_detection_raw(self, bbox, conf, img_bgr, lr):
        """Convert detection into standard format."""

        # pylint: disable=invalid-name, attribute-defined-outside-init
        if conf >= self.tracker_cfg.TRACK.CONFIDENCE_LOW:
            cx = bbox[0] + self.center_pos[0]
            cy = bbox[1] + self.center_pos[1]

            width = self.size[0] * (1 - lr) + bbox[2] * lr
            height = self.size[1] * (1 - lr) + bbox[3] * lr
        else:
            cx = self.center_pos[0]
            cy = self.center_pos[1]

            width = self.size[0]
            height = self.size[1]

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img_bgr.shape[:2])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        if conf < self.tracker_cfg.TRACK.CONFIDENCE_LOW:
            self.longterm_state = True
        elif conf > self.tracker_cfg.TRACK.CONFIDENCE_HIGH:
            self.longterm_state = False

        x1 = max(0, cx - width / 2)
        y1 = max(0, cy - height / 2)
        x2 = min(img_bgr.shape[1], cx + width / 2)
        y2 = min(img_bgr.shape[0], cy + height / 2)
        # pylint: enable=invalid-name, attribute-defined-outside-init
        det = np.array([[x1, y1, x2, y2, conf]]).astype('float32')
        return det

    def _get_best_bbox_raw(self, pred_bbox, score, window, scale_z):
        """Get best bbox."""

        def change(r):    # pylint: disable=invalid-name
            return np.maximum(r, 1. / r)

        def sz(w, h):   # pylint: disable=invalid-name
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
        # ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.tracker_cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window
        if not self.longterm_state:
            pscore = pscore * (1 - self.tracker_cfg.TRACK.WINDOW_INFLUENCE) + \
                    window * self.tracker_cfg.TRACK.WINDOW_INFLUENCE
        else:
            pscore = pscore * (1 - 0.001) + window * 0.001
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        # pylint: disable=invalid-name
        lr = penalty[best_idx] * score[best_idx] * self.tracker_cfg.TRACK.LR
        return bbox, score[best_idx], lr

    def detect_raw(self, img_bgr_list):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """

        if self.longterm_state:
            instance_size = self.tracker_cfg.TRACK.LOST_INSTANCE_SIZE
        else:
            instance_size = self.tracker_cfg.TRACK.INSTANCE_SIZE

        rough_det_queries = {}
        for query_instance_id in self.activated_query_list:
            det_galleries = []
            bbox_feats_z_half = self.bbox_feats_z_dic[query_instance_id]
            # self.model.zf = [item.float() for item in bbox_feats_z_half]
            self.model.zf = [torch.from_numpy(item).float().to('cuda') for item in bbox_feats_z_half]
            for img_bgr in img_bgr_list:
                scale_z, x_crop = self._get_scale_z_raw(instance_size, img_bgr)
                outputs = self.model.track(x_crop)
                score = self._convert_score(outputs['cls'])
                anchors, window = self._get_anchors(instance_size)
                pred_bbox = self._convert_bbox(outputs['loc'], anchors)    # (4, N)

                # pylint: disable=invalid-name
                bbox, conf, lr = self._get_best_bbox_raw(pred_bbox, score, window, scale_z)
                det = self._convert_detection_raw(bbox, conf, img_bgr, lr)
                det_galleries.append(det)
            rough_det_queries[query_instance_id] = det_galleries
        return rough_det_queries

    def detect_multi_query(self, img_bgr_list):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """

        if self.longterm_state:
            instance_size = self.tracker_cfg.TRACK.LOST_INSTANCE_SIZE
        else:
            instance_size = self.tracker_cfg.TRACK.INSTANCE_SIZE

        rough_det_queries = {}
        bbox_feats_z_list = []
        for query_instance_id in self.activated_query_list:
            bbox_feats_z = self.bbox_feats_z_dic[query_instance_id]
            bbox_feats_z_list.append(bbox_feats_z)
        # no query is responsible for this gallery, return empty dict
        if len(self.activated_query_list) == 0:
            return rough_det_queries
        self.model.zf = []
        for scale_index in range(len(bbox_feats_z_list[0])):
            sacle_feat = []
            for query_feat in bbox_feats_z_list:
                sacle_feat.append(query_feat[scale_index])
            # We ran out of gpu for large query data
            # mean_bbox_feats_z = torch.mean(torch.stack(sacle_feat), dim=0)
            # mean_bbox_feats_z = mean_bbox_feats_z.float()
            mean_bbox_feats_z = np.mean(sacle_feat, axis=0)
            mean_bbox_feats_z = torch.from_numpy(mean_bbox_feats_z).float().to('cuda')
            self.model.zf.append(mean_bbox_feats_z.cuda())
        batch_det_bboxes = []
        for img_bgr in img_bgr_list:
            scale_z, x_crop = self._get_scale_z_raw(instance_size, img_bgr)
            outputs = self.model.track(x_crop)
            score = self._convert_score(outputs['cls'])
            anchors, window = self._get_anchors(instance_size)
            pred_bbox = self._convert_bbox(outputs['loc'], anchors)    # (4, N)

            # pylint: disable=invalid-name
            bbox, conf, lr = self._get_best_bbox_raw(pred_bbox, score, window, scale_z)
            det = self._convert_detection_raw(bbox, conf, img_bgr, lr)
            batch_det_bboxes.append(det)
        for query_instance_id in self.activated_query_list:
            rough_det_queries[query_instance_id] = batch_det_bboxes
        return rough_det_queries
