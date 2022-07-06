# -*- coding:utf-8 -*-
"""Global tracker inference.
Author: gongyou.zyq
Date: 2020.11.18
"""

import os
import sys
import numpy as np
import cv2
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
sys.path.append('./instance_search/globaltracker/GlobalTrack')
sys.path.append('./instance_search/globaltracker/GlobalTrack/_submodules/mmdetection')
sys.path.append('./instance_search/globaltracker/GlobalTrack/_submodules/neuron')

# pylint: disable=wrong-import-position
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.cnn import normal_init
from neuron.models import Tracker
from mmdet.models.roi_extractors import SingleRoIExtractor
from mmdet.core import wrap_fp16_model, bbox2roi
from mmdet.models.detectors.two_stage import TwoStageDetector
from instance_search.base_localizer import BaseLocalizer
from instance_search.utils.io_utils import refine_bboxes
# pylint: enable=wrong-import-position

class RPN_Modulator(nn.Module):    # pylint: disable=invalid-name, abstract-method
    """Modify from GlobalTrack."""

    def __init__(self,
                 roi_out_size=7,
                 roi_sample_num=2,
                 channels=256,
                 strides=[4, 8, 16, 32],
                 featmap_num=5):
        # pylint: disable=dangerous-default-value, too-many-arguments
        super(RPN_Modulator, self).__init__()
        self.roi_extractor = SingleRoIExtractor(
            roi_layer={
                'type': 'RoIAlign',
                'out_size': roi_out_size,
                'sample_num': roi_sample_num},
            out_channels=channels,
            featmap_strides=strides)
        self.proj_modulator = nn.ModuleList([
            nn.Conv2d(channels, channels, roi_out_size, padding=0)
            for _ in range(featmap_num)])
        self.proj_out = nn.ModuleList([
            nn.Conv2d(channels, channels, 1, padding=0)
            for _ in range(featmap_num)])

    def custom_forward(self, modulator, feats_x):
        """Inference correlation. Very important!!!!!

        Args:
            modulator: bbox_feats_z, also known as modulator
            feats_x: 5 tuple of gallery feature map
        Returns:
            rpn_feats: 5 tuple of modulated feature map
        """

        n_imgs = len(feats_x[0])
        featmap_num = len(feats_x)
        query = modulator
        rpn_feats = []
        for k in range(featmap_num):
            batch_map = []
            for i in range(n_imgs):
                gallary = feats_x[k][i:i+1]
                proj_modulator = self.proj_modulator[k]
                proj_out = self.proj_out[k]
                out = proj_modulator(query) * gallary
                out = proj_out(out)
                batch_map.append(out)
            batch_map = torch.cat(batch_map, dim=0)
            rpn_feats.append(batch_map)
        return tuple(rpn_feats)

    def get_bbox_feats(self, feats_z, gt_bboxes_z):
        """Very important line, we should only run this once!!!!!"""

        # torch.Size([1, 5]), [class_type, x1, y1, x2, y2]
        rois = bbox2roi(gt_bboxes_z)
        # torch.Size([1, 256, 7, 7])
        bbox_feats_z = self.roi_extractor(
            feats_z[:self.roi_extractor.num_inputs], rois)
        # modulator = [bbox_feats[rois[:, 0] == j]
        #              for j in range(len(gt_bboxes_z))]
        return bbox_feats_z

    def init_weights(self):
        """Init weights."""

        for m in self.proj_modulator:    # pylint: disable=invalid-name
            normal_init(m, std=0.01)
        for m in self.proj_out:    # pylint: disable=invalid-name
            normal_init(m, std=0.01)


class RCNN_Modulator(nn.Module):    # pylint: disable=invalid-name
    """Modify from GlobalTrack."""

    def __init__(self, channels=256):
        super(RCNN_Modulator, self).__init__()
        self.proj_z = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj_x = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, z, x):    # pylint: disable=invalid-name, arguments-differ
        return self.inference(x, self.learn(z))

    def inference(self, x, modulator):    # pylint: disable=invalid-name
        """assume one image and one instance only"""

        assert len(modulator) == 1
        return self.proj_out(self.proj_x(x) * modulator)

    def learn(self, z):    # pylint: disable=invalid-name
        """assume one image and one instance only"""

        assert len(z) == 1
        return self.proj_z(z)

    def init_weights(self):
        """Init weights."""

        normal_init(self.proj_z, std=0.01)
        normal_init(self.proj_x, std=0.01)
        normal_init(self.proj_out, std=0.01)

class OpencvPreprocessor():
    """Opencv preprocessor."""

    def __init__(self, _cfg):
        # mean and std on gpu
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        self.mean = torch.from_numpy(mean).to("cuda").type(torch.float32)
        self.std = torch.from_numpy(std).to("cuda").type(torch.float32)
        self.scale = _cfg.INFERENCE.GLOBALTRACK_SCALE
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self._cfg = _cfg

    def resize_pad_image(self, img, img_meta):
        """Resize pad image."""

        h, w = img.shape[:2]    # pylint: disable=invalid-name
        scale = self.scale
        long_edge = max(scale)
        short_edge = min(scale)
        scale_factor = min(
            long_edge / max(h, w),
            short_edge / min(h, w))

        out_w = int(w * scale_factor + 0.5)
        out_h = int(h * scale_factor + 0.5)
        img = cv2.resize(img, (out_w, out_h), interpolation=1)
        img_meta.update({'img_shape': img.shape, 'scale_factor': scale_factor})

        # PadToDivisor
        divisor = 32
        out_h = int(np.ceil(img.shape[0] / divisor) * divisor)
        out_w = int(np.ceil(img.shape[1] / divisor) * divisor)
        shape = (out_h, out_w)
        if len(shape) < len(img.shape):
            shape += (img.shape[-1], )
        assert all([so >= si for so, si in zip(shape, img.shape)])
        pad_img = np.empty(shape, dtype=img.dtype)
        pad_img[...] = 0    # border_value=0
        pad_img[:img.shape[0], :img.shape[1], ...] = img
        img_meta.update({'pad_shape': pad_img.shape})
        return pad_img, img_meta, scale_factor

    def preprocess_gallery(self, img, img_meta, bboxes):
        """Opencv preprocess."""

        pad_img, img_meta, scale_factor = self.resize_pad_image(img, img_meta)
        # img_tensor = torch.from_numpy(pad_img.transpose(2, 0, 1)).float()
        img_tensor = torch.from_numpy(pad_img).to(self.device).type(torch.float32)
        img_tensor -= self.mean
        img_tensor /= self.std
        img_tensor = img_tensor.transpose(0, 2).transpose(1, 2)
        # img_tensor = img_tensor.transpose(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0).contiguous().to(self.device,\
                                                             non_blocking=True)
        img_meta.update({'flip': False})

        # modify bboxes
        if bboxes is not None:
            bboxes = bboxes * scale_factor
            pad_height, pad_width = pad_img.shape[:2]
            bboxes[..., 0::2] = np.clip(bboxes[..., 0::2], 0, pad_width - 1)
            bboxes[..., 1::2] = np.clip(bboxes[..., 1::2], 0, pad_height - 1)
            if bboxes.shape[1] == 4:
                bboxes[..., 2:] = np.clip(
                    bboxes[..., 2:], bboxes[..., :2], None)
            bboxes = torch.from_numpy(bboxes).to(self.device).type(torch.float32)
        return img_tensor, img_meta, bboxes

    def preprocess_query(self, large_img_bgr, query_bbox):
        """Preprocess query."""

        query_patch = large_img_bgr[query_bbox[1]: query_bbox[3],
                                    query_bbox[0]:query_bbox[2], :]
        small_height, small_width = query_patch.shape[:2]
        if self._cfg.EVAL.QUERY_FORMAT == 'large_bbox':
            large_img_rgb = cv2.cvtColor(large_img_bgr, cv2.COLOR_BGR2RGB)
            query_patch = large_img_rgb
        elif self._cfg.EVAL.QUERY_FORMAT == 'small_pure':
            query_bbox = np.array([0, 0, small_width, small_height])
            query_patch = cv2.cvtColor(query_patch, cv2.COLOR_BGR2RGB)
        elif self._cfg.EVAL.QUERY_FORMAT == 'small_pad':
            # pad_ratio = 0.5, no pad at all
            pad_ratio = 0.7
            fixed_height = small_height
        elif self._cfg.EVAL.QUERY_FORMAT == 'small_pad_context':
            pad_ratio = 2.0
            fixed_height = 500
        else:
            print('unkown query format')
        if self._cfg.EVAL.QUERY_FORMAT in ['small_pad', 'small_pad_context']:
            pad_info = {'pad_ratio': pad_ratio,
                        'fixed_height': fixed_height,
                        'new_large_height': 1080,
                        'new_large_width': 1920}
            padded_bbox_old = self._get_padded_bbox_old(large_img_bgr,\
                                                      query_bbox, pad_ratio)
            padded_height_before = padded_bbox_old[3]-padded_bbox_old[1]
            ratio = fixed_height / float(padded_height_before)
            blank = self._get_blank(ratio, padded_bbox_old, large_img_bgr, pad_info)
            new_query_bbox = self._get_query_bbox_new(query_bbox,
                                                      padded_bbox_old, ratio,
                                                      query_patch, pad_info)
            # cv2.rectangle(blank, (new_query_bbox[0], new_query_bbox[1]),
            #               (new_query_bbox[2], new_query_bbox[3]),
            #               (0, 0, 255), 2)
            # cv2.imwrite('debug.jpg', blank)
            blank = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)
            query_patch = blank
            query_bbox = new_query_bbox
        return query_patch, query_bbox

    @staticmethod
    def _get_query_bbox_new(query_bbox, padded_bbox_old, ratio, query_patch, pad_info):
        """Get new query bbox."""

        center_x_padded = (padded_bbox_old[0]+padded_bbox_old[2])/2.0
        center_y_padded = (padded_bbox_old[1]+padded_bbox_old[3])/2.0
        new_center_x = int(pad_info['new_large_width']/2)
        new_center_y = int(pad_info['new_large_height']/2)
        query_bbox_x1 = (query_bbox[0]-center_x_padded)*ratio+new_center_x
        query_bbox_y1 = (query_bbox[1]-center_y_padded)*ratio+new_center_y
        query_patch = cv2.resize(query_patch, (0, 0), fx=ratio, fy=ratio)
        new_query_bbox = [query_bbox_x1, query_bbox_y1,
                          query_bbox_x1+query_patch.shape[1],
                          query_bbox_y1+query_patch.shape[0]]
        new_query_bbox = np.array(new_query_bbox).astype('int')
        return new_query_bbox

    @staticmethod
    def _crop_patch(img, bbox):
        """Crop patch with given bbox."""

        [x1, y1, x2, y2] = bbox    # pylint: disable=invalid-name
        return img[y1: y2, x1: x2, :]

    def _get_blank(self, ratio, padded_bbox_old, large_img_bgr, pad_info_dic):
        """Get blank image."""

        padded_query_patch = self._crop_patch(large_img_bgr, padded_bbox_old)
        padded_query_patch = cv2.resize(padded_query_patch, (0, 0),
                                        fx=ratio, fy=ratio)
        new_large_height = pad_info_dic['new_large_height']
        new_large_width = pad_info_dic['new_large_width']
        blank = np.zeros((new_large_height, new_large_width, 3),
                         dtype=np.uint8)
        new_center_x = int(new_large_width/2)
        new_center_y = int(new_large_height/2)
        resized_x1 = int(new_center_x - padded_query_patch.shape[1] / 2)
        resized_y1 = int(new_center_y - padded_query_patch.shape[0] / 2)
        blank[resized_y1: resized_y1+padded_query_patch.shape[0],
              resized_x1: resized_x1+padded_query_patch.shape[1], :] = \
                                                            padded_query_patch
        return blank

    @staticmethod
    def _get_padded_bbox_old(large_img_bgr, query_bbox, pad_ratio):
        """Get padded bbox."""

        large_height, large_width = large_img_bgr.shape[:2]
        query_patch = large_img_bgr[query_bbox[1]: query_bbox[3],
                                    query_bbox[0]:query_bbox[2], :]
        small_height, small_width = query_patch.shape[:2]
        cx_query = int((query_bbox[0]+query_bbox[2])/2.0)
        cy_query = int((query_bbox[1]+query_bbox[3])/2.0)
        padded_x1 = int(max(0, cx_query-pad_ratio*small_width))
        padded_y1 = int(max(0, cy_query-pad_ratio*small_height))
        padded_x2 = int(min(large_width, cx_query+pad_ratio*small_width))
        padded_y2 = int(min(large_height, cy_query+pad_ratio*small_height))
        return np.array([padded_x1, padded_y1, padded_x2, padded_y2])

class QG_RCNN(TwoStageDetector):    # pylint: disable=invalid-name
    """Modify from GlobalTrack."""

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        # pylint: disable=too-many-arguments
        super(QG_RCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        self.rpn_modulator = RPN_Modulator()
        self.rcnn_modulator = RCNN_Modulator()
        self.rpn_modulator.init_weights()
        self.rcnn_modulator.init_weights()
        self._cfg = None

    def set_ins_config(self, _cfg):
        """Set instance search config."""

        self._cfg = _cfg

    # pylint: disable=invalid-name, too-many-arguments, too-many-locals, arguments-differ
    def simple_test_bboxes(self,
                           x,
                           img_meta_x,
                           proposals_list,
                           bbox_feats_z,
                           rcnn_test_cfg,
                           rescale=False):
        """Modify from GlobalTrack.

        Returns: A N-way list of batch det result, each element is (rcnn.max_per_img, 5)
        """

        _ = rescale
        # NOTE: No batch mode for proposals_list at this mmdetection version.
        batch_det_bboxes = []
        for proposals in proposals_list:

            # bbox head forward of gallary
            rois_x = bbox2roi([proposals])
            bbox_feats_x = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois_x)

            # do modulation    [1000, 256, 7, 7]
            roi_feats = self.rcnn_modulator(bbox_feats_z, bbox_feats_x)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            cls_score, bbox_pred = self.bbox_head(roi_feats)

            # get predictions
            img_shape = img_meta_x[0]['img_shape']
            scale_factor = img_meta_x[0]['scale_factor']

            det_bboxes, _ = self.bbox_head.get_det_bboxes(
                rois_x,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=True,
                cfg=rcnn_test_cfg)
            rough_det = det_bboxes.cpu().numpy().astype('float16')
            good_data_mask = refine_bboxes(rough_det, self._cfg)
            rough_det = rough_det[good_data_mask]
            batch_det_bboxes.append(rough_det)
        return batch_det_bboxes


class GlobalTracker(Tracker, BaseLocalizer):    # pylint: disable=abstract-method
    """Global Tracker."""

    def __init__(self, gpus, _cfg):

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
        BaseLocalizer.__init__(self, gpus, _cfg)

        name = 'GlobalTrack'
        name_suffix = 'qg_rcnn_r50_fpn'
        if name_suffix:
            name += '_' + name_suffix
        super(GlobalTracker, self).__init__(
            name=name, is_deterministic=True)
        self.half_flag = _cfg.INFERENCE.LOC_HALF_FLAG

        # pylint: disable=line-too-long
        cfg_file = './instance_search/globaltracker/GlobalTrack/configs/qg_rcnn_r50_fpn.py'
        ckp_file = _cfg.INFERENCE.LOCALIZER_MODEL
        # ckp_file = '/home/gongyou.zyq/GlobalTrack/work_dirs/qg_rcnn_r50_fpn_baseline/latest.pth'
        # pylint: enable=line-too-long
        global_tracker_cfg = Config.fromfile(cfg_file)
        # if global_tracker_cfg.get('cudnn_benchmark', False):
        #     torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark = False
        global_tracker_cfg.model.pretrained = None

        # build model
        # global_tracker_cfg.test_cfg.rpn.nms_pre = 300
        # global_tracker_cfg.test_cfg.rpn.nms_post = 300
        # global_tracker_cfg.test_cfg.rpn.max_num = 3
        # global_tracker_cfg.test_cfg.rpn.nms_thr = _cfg.EVAL.NMS
        # global_tracker_cfg.test_cfg.rpn.min_bbox_size = _cfg.EVAL.MIN_HEIGHT/2
        global_tracker_cfg.test_cfg.rcnn.max_per_img = _cfg.INFERENCE.LOCALIZER_TOPK
        global_tracker_cfg.test_cfg.rcnn.nms.iou_thr = _cfg.EVAL.NMS
        self.model = self._build_from_cfg(global_tracker_cfg.model,
                                          test_cfg=global_tracker_cfg.test_cfg)
        self.model.set_ins_config(_cfg)

        if self.half_flag:
            wrap_fp16_model(self.model)
        load_checkpoint(self.model, ckp_file, map_location='cpu')
        self.model.CLASSES = ('object', )

        # GPU usage
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

        self.preprocessor = OpencvPreprocessor(_cfg)
        self.template_mean_feats_z = None
        self.cluster_feats_z = {}
        self.query2cluster = {}

    @staticmethod
    def _build_from_cfg(tracker_model_cfg, test_cfg=None):
        """Build a module from config dict.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
            registry (:obj:`Registry`): The registry to search the type from.
            default_args (dict, optional): Default initialization arguments.

        Returns:
            obj: The constructed object.
        """
        assert isinstance(tracker_model_cfg, dict) and 'type' in tracker_model_cfg
        assert isinstance(test_cfg, dict)
        default_args = dict(train_cfg=None, test_cfg=test_cfg)
        args = tracker_model_cfg.copy()
        obj_type = args.pop('type')
        assert obj_type == 'QG_RCNN'
        for name, value in default_args.items():
            args.setdefault(name, value)
        return QG_RCNN(**args)

    @torch.no_grad()
    def set_query(self, query_img_bgr, query_bbox, query_instance_id):
        """Set query."""

        query_patch, query_bbox = self.preprocessor.preprocess_query(query_img_bgr, query_bbox)
        img = query_patch
        bbox = query_bbox

        # prepare query data
        img_meta = {'ori_shape': img.shape}
        bboxes = np.expand_dims(bbox, axis=0)
        img, img_meta, bboxes = self.preprocessor.preprocess_gallery(img, img_meta, bboxes)

        # initialize the modulator
        if self.half_flag:
            img = img.half()
        bbox_feats_z = self.process_query(img, [bboxes])
        self.bbox_feats_z_dic[query_instance_id] = bbox_feats_z.cpu().numpy()
        return self.bbox_feats_z_dic

    def process_query(self, img_z, gt_bboxes_z):
        """Extract feature map and set query_bbox, no roi operations here."""

        # img_z: (768, 1344), feats_z: fpn feature in 5 tuples
        # torch.Size([1, 256, 192, 336]) torch.Size([1, 256, 96, 168])
        # torch.Size([1, 256, 48, 84]) torch.Size([1, 256, 24, 42])
        # torch.Size([1, 256, 12, 21])
        feats_z = self.model.extract_feat(img_z)
        # # Just crop roi_z, (1, 256, 7, 7)
        bbox_feats_z = self.model.rpn_modulator.get_bbox_feats(feats_z,
                                                               gt_bboxes_z)
        return bbox_feats_z

    def process_gallery_memory(self, img_x, img_meta_x):
        """Extract gallery feature map and more op here."""

        x = self.model.extract_feat(img_x)    # pylint: disable=invalid-name

        rough_det_queries = []
        featmap_num = len(x)
        rpn_feats_queries = [[] for _ in range(featmap_num)]
        for i in range(self.query_num):
            bbox_feats_z = self.bbox_feats_z_list[i]
            rpn_feats = self.model.rpn_modulator.custom_forward(bbox_feats_z,
                                                                x)
            for k in range(featmap_num):
                rpn_feats_queries[k].append(rpn_feats[k])

        mean_rpn_feats = []
        for k in range(featmap_num):
            mean_feats = torch.mean(torch.stack(rpn_feats_queries[k]),
                                    dim=0)
            mean_rpn_feats.append(mean_feats)
        mean_rpn_feats = tuple(mean_rpn_feats)
        proposal_list = self.model.simple_test_rpn(
            mean_rpn_feats, img_meta_x, self.model.test_cfg.rpn)
        for i in range(self.query_num):
            bbox_feats_z = self.bbox_feats_z_list[i]
            batch_det_bboxes = self.model.simple_test_bboxes(x, img_meta_x,
                                                             proposal_list,
                                                             bbox_feats_z,
                                                             self.model.test_cfg.rcnn)
            rough_det_queries.append(batch_det_bboxes)
        return rough_det_queries

    def get_mean_template(self):
        """Get mean template.
        Only used in global gallery search
        """

        bbox_feats_z_list = []
        for bbox_feats_z in self.bbox_feats_z_dic.values():
            bbox_feats_z_list.append(bbox_feats_z)
        mean_bbox_feats_z = torch.mean(torch.stack(bbox_feats_z_list),
                                       dim=0)
        self.template_mean_feats_z = mean_bbox_feats_z

    def cluster_template(self):
        """Get clustered template.
        Only used in global gallery search
        """

        feats_list = []
        cluster_num = 50
        query_id_list = list(self.bbox_feats_z_dic.keys())
        for bbox_feats_z in self.bbox_feats_z_dic.values():
            feats_list.append(bbox_feats_z.cpu().numpy().flatten())
        feats_list = np.array(feats_list)
        kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(feats_list)
        for query_instance_id, cluster_id in zip(query_id_list, kmeans.labels_):
            self.query2cluster[query_instance_id] = cluster_id
        for cluster_id, cluster_center in enumerate(kmeans.cluster_centers_):
            cluster_center = cluster_center.reshape((1, 256, 7, 7))
            self.cluster_feats_z[cluster_id] = \
                    torch.from_numpy(cluster_center).half().to('cuda')

    def process_gallery_multi_query(self, img_x, img_meta_x):
        """Extract gallery feature map and more op here."""

        x = self.model.extract_feat(img_x)    # pylint: disable=invalid-name

        rough_det_queries = {}
        if self._cfg.EVAL.GALLERY_RANGE == 'global' and\
                self.template_mean_feats_z is not None:
            mean_bbox_feats_z = self.template_mean_feats_z
        else:
            bbox_feats_z_list = []
            for query_instance_id in self.activated_query_list:
                bbox_feats_z = self.bbox_feats_z_dic[query_instance_id]
                bbox_feats_z_list.append(bbox_feats_z)
            # no query is responsible for this gallery, return empty dict
            if len(self.activated_query_list) == 0:
                return rough_det_queries
            # mean_bbox_feats_z = torch.mean(torch.stack(bbox_feats_z_list),
            #                                dim=0)
            mean_bbox_feats_z = torch.mean(torch.stack([torch.tensor(item).cuda() for item in bbox_feats_z_list]),
                                           dim=0)
            self.template_mean_feats_z = mean_bbox_feats_z

        rpn_feats = self.model.rpn_modulator.custom_forward(mean_bbox_feats_z, x)
        proposal_list = self.model.simple_test_rpn(
            rpn_feats, img_meta_x, self.model.test_cfg.rpn)
        batch_det_bboxes = self.model.simple_test_bboxes(x, img_meta_x, proposal_list,\
                                            mean_bbox_feats_z, self.model.test_cfg.rcnn)
        for query_instance_id in self.activated_query_list:
            rough_det_queries[query_instance_id] = batch_det_bboxes
        return rough_det_queries

    def process_gallery(self, img_x, img_meta_x):
        """Extract gallery feature map and more op here."""

        if self._cfg.EVAL.MULTI_QUERY:
            return self.process_gallery_multi_query(img_x, img_meta_x)
        else:
            return self.process_gallery_single_query(img_x, img_meta_x)
        # return self.process_gallery_share_rpn(img_x, img_meta_x)
        # return self.process_gallery_cluster(img_x, img_meta_x)

    def process_gallery_single_query(self, img_x, img_meta_x):
        """Extract gallery feature map and more op here."""

        x = self.model.extract_feat(img_x)    # pylint: disable=invalid-name

        rough_det_queries = {}
        for query_instance_id in self.activated_query_list:
            bbox_feats_z = self.bbox_feats_z_dic[query_instance_id]

            # RPN forward, self-defined op
            rpn_feats = self.model.rpn_modulator.custom_forward(bbox_feats_z, x)
            # (N, 256, 200, 256), (N, 256, 100, 128), (N, 256, 50, 64)
            # (N, 256, 25, 32), (N, 256, 13, 16)
            # common op from mmdetection
            # proposal_list: A list of N, each element is (1000, 5)
            # Can not ba accelerated, very slow
            proposal_list = self.model.simple_test_rpn(
                rpn_feats, img_meta_x, self.model.test_cfg.rpn)

            # RCNN forward, self modified op
            batch_det_bboxes = self.model.simple_test_bboxes(x, img_meta_x, proposal_list,
                                                             bbox_feats_z, self.model.test_cfg.rcnn)
            rough_det_queries[query_instance_id] = batch_det_bboxes
        return rough_det_queries

    def process_gallery_share_rpn(self, img_x, img_meta_x):
        """Extract gallery feature map and more op here."""

        x = self.model.extract_feat(img_x)    # pylint: disable=invalid-name

        rough_det_queries = {}
        if self._cfg.EVAL.GALLERY_RANGE == 'global' and\
                self.template_mean_feats_z is not None:
            mean_bbox_feats_z = self.template_mean_feats_z
        else:
            bbox_feats_z_list = []
            for query_instance_id in self.activated_query_list:
                bbox_feats_z = self.bbox_feats_z_dic[query_instance_id]
                bbox_feats_z_list.append(bbox_feats_z)
            # no query is responsible for this gallery, return empty dict
            if len(self.activated_query_list) == 0:
                return rough_det_queries
            mean_bbox_feats_z = torch.mean(torch.stack(bbox_feats_z_list),
                                           dim=0)
            self.template_mean_feats_z = mean_bbox_feats_z

        rpn_feats = self.model.rpn_modulator.custom_forward(mean_bbox_feats_z, x)
        proposal_list = self.model.simple_test_rpn(
            rpn_feats, img_meta_x, self.model.test_cfg.rpn)
        for query_instance_id in self.activated_query_list:
            batch_det_bboxes = self.model.simple_test_bboxes(x, img_meta_x,\
                    proposal_list, self.bbox_feats_z_dic[query_instance_id],\
                    self.model.test_cfg.rcnn)
            rough_det_queries[query_instance_id] = batch_det_bboxes
        return rough_det_queries

    def process_gallery_cluster(self, img_x, img_meta_x):
        """Extract gallery feature map and more op here."""

        x = self.model.extract_feat(img_x)    # pylint: disable=invalid-name

        cluster_ret = {}
        for cluster_id, cluster_center in self.cluster_feats_z.items():
            rpn_feats = self.model.rpn_modulator.custom_forward(cluster_center, x)
            proposal_list = self.model.simple_test_rpn(
                rpn_feats, img_meta_x, self.model.test_cfg.rpn)
            batch_det_bboxes = self.model.simple_test_bboxes(x, img_meta_x,\
                    proposal_list, cluster_center, self.model.test_cfg.rcnn)
            cluster_ret[cluster_id] = batch_det_bboxes

        rough_det_queries = {}
        for query_instance_id in self.activated_query_list:
            cluster_id = self.query2cluster[query_instance_id]
            rough_det_queries[query_instance_id] = cluster_ret[cluster_id]
        return rough_det_queries

    @torch.no_grad()
    def detect(self, img_bgr_list):
        """Detect global track.
        Args:
            img_bgr_list: A list of opencv image as batch input.

        Returns:
            det_queries: Multi query+multi gallery detection results.
                         e.g. results[query_index][gallery_index]: detection
        """

        img_list = []
        img_meta_list = []
        for img_bgr in img_bgr_list:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # prepare gallary data
            img_meta = {'ori_shape': img.shape}
            # re-implementation, norm in gpu, 2ms
            img, img_meta, _ = self.preprocessor.preprocess_gallery(img, img_meta, None)
            img_list.append(img)
            img_meta_list.append(img_meta)
        img = torch.cat(img_list, 0)

        # get detections
        if self.half_flag:
            img = img.half()
        # start = time.time()
        det_queries = self.process_gallery(img, img_meta_list)
        # print(f'extract time {time.time()-start}')
        return det_queries
