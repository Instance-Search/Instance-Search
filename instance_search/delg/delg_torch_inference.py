"""Extract global-reid features in the same format as delg.
Author: gongyou.zyq
Date: 2021.06.10
"""
import os
import sys
sys.path.append('/home/gongyou.zyq/ILR2021/submission/util-code')

import cv2
import numpy as np
import torch

from instance_search.base_matcher import BaseMatcher
from make_model import make_model

class ReIDInference:
    """ReID Inference."""

    def __init__(self, gpus, _cfg):

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
        self.mean = np.array([123.675, 116.280, 103.530])
        self.std = np.array([57.0, 57.0, 57.0])
        self.mean_gpu = torch.tensor(self.mean).cuda()
        self.std_gpu = torch.tensor(self.std).cuda()

        IMAGE_SIZE = 256
        self.model = make_model('efficientnet-b0')
        # self.model.load_param('/home/gongyou.zyq/ILR2021/submission/models/efficientnet-b7_30.pth')
        self.model.load_param('/home/gongyou.zyq/ILR2021/logs/GLDv2clean/CNN/efficientnet-b0_10.pth')

        # IMAGE_SIZE = 256
        # self.model = make_model('resnet50')
        # self.model.load_param('/home/gongyou.zyq/ILR2021/submission/models/R50_256.pth')

        # IMAGE_SIZE = 768
        # self.model = make_model('resnest269')
        # self.model.load_param('/home/gongyou.zyq/ILR2021/logs/GLDv2clean/ResNeSt269_448_finetune/resnest269_10.pth')

        self.model.to('cuda')
        self.model.eval()
        self.image_width = IMAGE_SIZE
        self.image_height = IMAGE_SIZE

    def extract(self, img):
        """Extract feature for one image."""

        img = cv2.resize(img, (self.image_width, self.image_height))
        img = torch.tensor(img).cuda()
        img = img[:, :, [2, 1, 0]]
        img = (img-self.mean_gpu)/self.std_gpu
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).float()
        feat = self.model(img).cpu().detach().numpy().flatten()
        feat = feat/np.linalg.norm(feat, ord=2, axis=0)
        return feat

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
            patch = torch.tensor(patch).cuda().float()
            feat = self.model(patch).cpu().detach().numpy().flatten()
            feat = feat/np.linalg.norm(feat, ord=2, axis=0)
            feat_list.append(feat)
        feat_list = np.vstack(feat_list)
        return feat_list

class DelgTorchMatcher(BaseMatcher):
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
