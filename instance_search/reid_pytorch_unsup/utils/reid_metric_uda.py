# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking

import sklearn.metrics.cluster
from sklearn.cluster import DBSCAN
from collections import Counter

def ComputeEuclid(array1,array2,fg_sqrt=True):
    #array1:[m1,n],array2:[m2,n]
    assert array1.shape[1]==array2.shape[1];
    # shape [m1, 1]
    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    # shape [1, m2]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    #print 'array1,array2 shape:',array1.shape,array2.shape
    squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
    squared_dist[squared_dist < 0] = 0
    #shape [m1,m2]
    if fg_sqrt:
        dist = np.sqrt(squared_dist)
        #print('[test] using sqrt for distance')
    else:
        dist = squared_dist
        #print('[test] not using sqrt for distance')
    return dist**2/2.

class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            #print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]

        #distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #          torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        #distmat.addmm_(1, -2, qf, gf.t())
        #distmat = distmat.cpu().numpy()

        qf_np = qf.cpu().numpy().astype(np.float32)
        gf_np = gf.cpu().numpy().astype(np.float32)
        #a = np.repeat(np.sum(np.power(qf_np, 2), axis=1, keepdims=True), n, axis=1)
        #b = np.repeat(np.transpose(np.sum(np.power(gf_np, 2), axis=1, keepdims=True)), m, axis=0)
        #distmat = a + b - 2 * np.matmul(qf_np, np.transpose(gf_np))

        distmat = ComputeEuclid(qf_np,gf_np)

        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            #print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP

class Cluster(Metric):
    def __init__(self, config, topk, dist_thrd, min_samples, feat_norm='yes'):
        super(Cluster, self).__init__()
        self.feat_norm = feat_norm
        self.topk = topk
        self.dist_thrd = dist_thrd
        self.min_samples = min_samples
        self._cfg = config

    def reset(self):
        self.feats = []
        self.pids = []
        self.raw_image_names = []
        self.bboxes = []

    def update(self, output):
        feat, pid, camid, trkid, raw_image_name, bbox = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.raw_image_names.extend(raw_image_name)
        self.bboxes.extend(bbox)

    @staticmethod
    def compute_iou(bb_test, bb_gt):
        """
        Computes IUO between two bboxes in the form [x1,y1,x2,y2]
        """

        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        # pylint: disable=invalid-name
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
                  + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
        # pylint: enable=invalid-name
        return o

    def compute_iou_distmat(self):
        """Compute pair-wise iou among data. Small value means similar"""

        raw_image_names = np.array(self.raw_image_names, dtype=object)
        bboxes = np.array(self.bboxes).astype('float32')
        same_image_flag = np.equal(raw_image_names[:, np.newaxis],
                                   raw_image_names[np.newaxis, :])
        same_image_flag = same_image_flag.astype('float32')
        same_pos = np.where(same_image_flag > 0)
        for (i, j) in zip(same_pos[0], same_pos[1]):
            bbox_i = bboxes[i]
            bbox_j = bboxes[j]
            same_image_flag[i, j] = self.compute_iou(bbox_i, bbox_j)
        iou_distmat = 1.0 - same_image_flag
        return iou_distmat

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            #print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        feats = feats.cpu().numpy().astype(np.float32)
        visual_distmat = ComputeEuclid(feats, feats)    # 0.0-2.0
        # distmat = (1.0 - self._cfg.TGT_UNSUPDATA.IOU_WEIGHT) * visual_distmat + \
        #           self._cfg.TGT_UNSUPDATA.IOU_WEIGHT * iou_distmat
        if self._cfg.TGT_UNSUPDATA.IOU_WEIGHT > 0:
            assert self.bboxes[0] is not None    # bbox_unsup
            iou_distmat = self.compute_iou_distmat()
            distmat = visual_distmat + iou_distmat
        else:
            distmat = visual_distmat

        eps = self.dist_thrd
        cluster = DBSCAN(eps=eps, min_samples=self.min_samples, metric='precomputed', n_jobs=2)
        ret = cluster.fit_predict(distmat)

        # most count
        labelset = Counter(ret[ret >= 0])
        labelset = labelset.most_common()
        labelset = [i[0] for i in labelset]

        labelset = labelset[:self.topk]
        idxs = np.where(np.in1d(ret, labelset))[0]
        psolabels = ret[idxs]
        pids = np.array(self.pids)[idxs]
        acc = sklearn.metrics.cluster.normalized_mutual_info_score(psolabels, pids)
        outret = ret

        psofeatures = feats[idxs]
        mean_features = []
        for label in labelset:
            mean_indices = (psolabels == label)
            mean_features.append(np.mean(psofeatures[mean_indices], axis=0))
        mean_features = np.array(mean_features)
        # device = self.feats.get_device()
        mean_features = torch.tensor(mean_features)

        # relabel
        outret_set = set(outret)
        pid2label = {}
        for pid in outret_set:
            if pid in labelset:
                pid2label[pid] = labelset.index(pid)
            else:
                pid2label[pid] = -1
        outret = np.asarray([pid2label[i] for i in outret])

        return outret, acc, len(outret[outret != -1]), mean_features
